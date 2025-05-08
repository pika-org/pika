//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_xxx.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/mpi_helpers.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/debugging/demangle_helper.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution/algorithms/continues_on.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/just.hpp>
#include <pika/execution/algorithms/let_value.hpp>
#include <pika/execution/algorithms/unpack.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/mpi_base/mpi_exception.hpp>
#include <pika/synchronization/condition_variable.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::dispatch_mpi_detail {
    namespace ex = execution::experimental;
    namespace mpi = pika::mpi::experimental;

    // -----------------------------------------------------------------
    // operation state for an internal receiver
    template <typename Receiver, typename F, typename Sender>
    struct operation_state
    {
        PIKA_NO_UNIQUE_ADDRESS Receiver r;
        PIKA_NO_UNIQUE_ADDRESS F f;

        int mode_flags;
        int status;

        // these vars are needed by suspend/resume mode
        bool completed{false};
        pika::detail::spinlock mutex;
        pika::condition_variable cond_var;

        MPI_Request request{MPI_REQUEST_NULL};

        // -----------------------------------------------------------------
        // The mpi_receiver receives inputs from the previous sender,
        // invokes the mpi call, and sets a callback on the polling handler
        struct receiver
        {
            PIKA_STDEXEC_RECEIVER_CONCEPT

            operation_state& op_state;

            template <typename Error>
            constexpr void set_error(Error&& error) && noexcept
            {
                auto r = std::move(*this);
                ex::set_error(std::move(r.op_state.r), std::forward<Error>(error));
            }

            constexpr void set_stopped() && noexcept
            {
                auto r = std::move(*this);
                ex::set_stopped(std::move(r.op_state.r));
            }

            // receive the MPI function invocable + arguments and add a request,
            // then invoke the mpi function with the added request
            // if the invocation gives an error, set_error
            // otherwise return the request by passing it to set_value
            template <typename... Ts,
                typename = std::enable_if_t<mpi::detail::is_mpi_request_invocable_v<F, Ts...>>>
            constexpr void set_value(Ts&... ts) && noexcept
            {
                auto r = std::move(*this);
                pika::detail::try_catch_exception_ptr(
                    [&]() mutable {
                        using invoke_result_type =
                            mpi::detail::mpi_request_invoke_result_t<F, Ts...>;

                        PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>,
                            debug(str<>("dispatch_mpi_recv"), "set_value_t"));
#ifdef PIKA_HAVE_APEX
                        apex::scoped_timer apex_post("pika::mpi::post");
#endif
                        int status = MPI_SUCCESS;
                        // execute the mpi function call, passing in the request object
                        if constexpr (std::is_void_v<invoke_result_type>)
                        {
                            PIKA_INVOKE(std::move(r.op_state.f), ts..., &r.op_state.request);
                        }
                        else
                        {
                            static_assert(std::is_same_v<invoke_result_type, int>);
                            status =
                                PIKA_INVOKE(std::move(r.op_state.f), ts..., &r.op_state.request);
                        }
                        PIKA_DETAIL_DP(mpi::detail::mpi_tran<7>,
                            debug(
                                str<>("dispatch_mpi_recv"), "invoke mpi", ptr(r.op_state.request)));

                        PIKA_ASSERT_MSG(r.op_state.request != MPI_REQUEST_NULL,
                            "MPI_REQUEST_NULL returned from mpi invocation");

                        if (status != MPI_SUCCESS)
                        {
                            PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>,
                                debug(str<>("set_error"), "status != MPI_SUCCESS",
                                    pika::mpi::detail::error_message(status)));
                            ex::set_error(std::move(r.op_state.r),
                                std::make_exception_ptr(
                                    pika::mpi::exception(status, "dispatch mpi")));
                            return;
                        }

                        // early poll just in case the request completed immediately
                        if (mpi::detail::poll_request(r.op_state.request))
                        {
#ifdef PIKA_HAVE_APEX
                            apex::scoped_timer apex_invoke("pika::mpi::trigger");
#endif
                            PIKA_DETAIL_DP(mpi::detail::mpi_tran<7>,
                                debug(str<>("trigger_mpi_recv"), "eager poll ok",
                                    ptr(r.op_state.request)));
                            ex::set_value(std::move(r.op_state.r));
                            return;
                        }

                        // which polling/testing mode are we using
                        mpi::detail::handler_method mode =
                            mpi::detail::get_handler_method(r.op_state.mode_flags);
                        execution::thread_priority p =
                            mpi::detail::use_priority_boost(r.op_state.mode_flags) ?
                            execution::thread_priority::boost :
                            execution::thread_priority::normal;

                        PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>,
                            debug(str<>("trigger_mpi_recv"), "set_value_t", "req",
                                ptr(r.op_state.request), "flags", bin<8>(r.op_state.mode_flags),
                                mode_string(r.op_state.mode_flags)));

                        switch (mode)
                        {
                        case mpi::detail::handler_method::yield_while:
                        {
                            // yield/while is invalid on a non pika thread
                            PIKA_ASSERT(pika::threads::detail::get_self_id());
                            pika::util::yield_while(
                                [&r]() { return !mpi::detail::poll_request(r.op_state.request); },
                                "trigger_mpi wait for request");
#ifdef PIKA_HAVE_APEX
                            apex::scoped_timer apex_invoke("pika::mpi::trigger");
#endif
                            // we just assume the return from mpi_test is always MPI_SUCCESS
                            ex::set_value(std::move(r.op_state.r));
                            break;
                        }
                        case mpi::detail::handler_method::suspend_resume:
                        {
                            // suspend is invalid on a non pika thread
                            PIKA_ASSERT(pika::threads::detail::get_self_id());
                            // the callback will resume _this_ thread
                            {
                                std::unique_lock l{r.op_state.mutex};
                                mpi::detail::add_suspend_resume_request_callback(r.op_state);
                                if (mpi::detail::use_priority_boost(r.op_state.mode_flags))
                                {
                                    threads::detail::thread_data::scoped_thread_priority
                                        set_restore(p);
                                    r.op_state.cond_var.wait(
                                        l, [&]() { return r.op_state.completed; });
                                }
                                else
                                {
                                    r.op_state.cond_var.wait(
                                        l, [&]() { return r.op_state.completed; });
                                }
                            }

#ifdef PIKA_HAVE_APEX
                            apex::scoped_timer apex_invoke("pika::mpi::trigger");
#endif
                            // call set_value/set_error depending on mpi return status
                            mpi::detail::set_value_error_helper(
                                r.op_state.status, std::move(r.op_state.r));
                            break;
                        }
                        case mpi::detail::handler_method::new_task:
                        {
                            // The callback will call set_value/set_error inside a new task
                            // and execution will continue on that thread
                            mpi::detail::add_new_task_request_callback(r.op_state);
                            break;
                        }
                        case mpi::detail::handler_method::continuation:
                        {
                            // The callback will call set_value/set_error
                            // execution will continue on the callback thread
                            mpi::detail::add_continuation_request_callback(r.op_state);
                            break;
                        }
                        case mpi::detail::handler_method::mpix_continuation:
                        {
                            PIKA_DETAIL_DP(mpi::detail::mpi_tran<1>,
                                debug(str<>("MPI_EXT_CONTINUE"), "register_mpix_continuation",
                                    ptr(r.op_state.request), ptr(r.op_state.request)));
                            mpi::detail::MPIX_Continue_cb_function* func =
                                &mpi::detail::mpix_callback_continuation<operation_state>;
                            mpi::detail::register_mpix_continuation(
                                &r.op_state.request, func, &r.op_state);
                            break;
                        }
                        default: PIKA_UNREACHABLE;
                        }
                    },
                    [&](std::exception_ptr ep) {
                        ex::set_error(std::move(r.op_state.r), std::move(ep));
                    });
            }

            constexpr ex::empty_env get_env() const& noexcept { return {}; }
        };

        using operation_state_type = ex::connect_result_t<Sender, receiver>;
        operation_state_type op_state;

        template <typename Receiver_, typename F_, typename Sender_>
        operation_state(Receiver_&& r, F_&& f, int flags, Sender_&& sender)
          : r(std::forward<Receiver_>(r))
          , f(std::forward<F_>(f))
          , mode_flags{flags}
          , op_state(ex::connect(std::forward<Sender_>(sender), receiver{*this}))
        {
            PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>, debug(str<>("operation_state")));
        }

        void start() & noexcept { return ex::start(op_state); }
    };

    // -----------------------------------------------------------------
    // transform MPI adapter - sender type
    template <typename Sender, typename F>
    struct sender
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        PIKA_NO_UNIQUE_ADDRESS Sender sender;
        PIKA_NO_UNIQUE_ADDRESS F f;
        int completion_mode_flags;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename...>
        using no_value_completion = pika::execution::experimental::completion_signatures<>;
        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<Sender,
                pika::execution::experimental::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_value_t(),
                    pika::execution::experimental::set_error_t(std::exception_ptr)>,
                no_value_completion>;
#else
        // -----------------------------------------------------------------
        // completion signatures
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <typename...> class Variant>
        using error_types = util::detail::unique_t<util::detail::prepend_t<
            typename ex::sender_traits<Sender>::template error_types<Variant>, std::exception_ptr>>;

#endif

        static constexpr bool sends_done = false;

        template <typename Receiver>
        constexpr auto connect(Receiver&& receiver) const&
        {
            return operation_state<std::decay_t<Receiver>, F, Sender>(
                std::forward<Receiver>(receiver), f, completion_mode_flags, sender);
        }

        template <typename Receiver>
        constexpr auto connect(Receiver&& receiver) &&
        {
            return operation_state<std::decay_t<Receiver>, F, Sender>(
                std::forward<Receiver>(receiver), std::move(f), completion_mode_flags,
                std::move(sender));
        }
    };

}    // namespace pika::dispatch_mpi_detail

namespace pika::mpi::experimental {
    inline constexpr struct dispatch_mpi_t final
      : pika::functional::detail::tag_fallback<dispatch_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(dispatch_mpi_t, Sender&& sender, F&& f, int flags)
        {
            return dispatch_mpi_detail::sender<std::decay_t<Sender>, std::decay_t<F>>{
                std::forward<Sender>(sender), std::forward<F>(f), flags};
        }

        // TODO: flags -> mode, int -> std::size_t
        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(dispatch_mpi_t, F&& f, int flags)
        {
            return pika::execution::experimental::detail::partial_algorithm<dispatch_mpi_t, F, int>{
                std::forward<F>(f), flags};
        }
    } dispatch_mpi{};

    inline constexpr struct transform_mpi_t final
      : pika::functional::detail::tag_fallback<transform_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<std::decay_t<Sender>>)>
        friend PIKA_FORCEINLINE pika::execution::experimental::unique_any_sender<>
        tag_fallback_invoke(transform_mpi_t, Sender&& sender, F&& f)
        {
            using namespace pika::mpi::experimental::detail;
            PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("transform_mpi_t"), "tag_fallback_invoke"));

            using execution::thread_priority;
            using pika::execution::experimental::continues_on;
            using pika::execution::experimental::just;
            using pika::execution::experimental::let_value;
            using pika::execution::experimental::unique_any_sender;
            using pika::execution::experimental::unpack;

            // get mpi completion mode settings
            auto mode = get_completion_mode();
            bool completions_inline = use_inline_completion(mode);
            bool requests_inline = use_inline_request(mode);

            execution::thread_priority p = use_priority_boost(mode) ?
                execution::thread_priority::boost :
                execution::thread_priority::normal;

            auto f_completion = [f = std::forward<F>(f), mode, completions_inline, p](
                                    auto&... args) mutable -> unique_any_sender<> {
                unique_any_sender<> s = just(std::forward_as_tuple(args...)) | unpack() |
                    dispatch_mpi(std::move(f), mode);
                if (completions_inline) { return s; }
                else { return std::move(s) | continues_on(default_pool_scheduler(p)); }
            };

            if (requests_inline)
            {
                return std::forward<Sender>(sender) | let_value(std::move(f_completion));
            }
            else
            {
                return std::forward<Sender>(sender) | continues_on(mpi_pool_scheduler(p)) |
                    let_value(std::move(f_completion));
            }
        }

        //
        // tag invoke overload for mpi_transform
        //
        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(transform_mpi_t, F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<transform_mpi_t, F>{
                std::forward<F>(f)};
        }

    } transform_mpi{};
}    // namespace pika::mpi::experimental
