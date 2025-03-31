//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_xxx.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/dispatch_mpi.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/debugging/demangle_helper.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/synchronization/condition_variable.hpp>

#include <chrono>
#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::trigger_mpi_detail {
    namespace ex = pika::execution::experimental;
    namespace mpi = pika::mpi::experimental;

    // -----------------------------------------------------------------
    // operation state for an internal receiver
    template <typename Receiver, typename Sender>
    struct operation_state
    {
        PIKA_NO_UNIQUE_ADDRESS Receiver r;
        int mode_flags;
        int status;
        // these vars are needed by suspend/resume mode
        bool completed{false};
        pika::detail::spinlock mutex{};
        pika::condition_variable cond_var{};
        // MPI_EXT_CONTINUE
        MPI_Request request{MPI_REQUEST_NULL};
        std::chrono::duration<double> eager_poll_busy_wait_timeout;

        // -----------------------------------------------------------------
        // The mpi_receiver receives inputs from the previous sender,
        // invokes the mpi call, and sets a callback on the polling handler
        struct receiver
        {
            PIKA_STDEXEC_RECEIVER_CONCEPT
            operation_state& op_state;

            template <typename Error>
            friend constexpr void tag_invoke(ex::set_error_t, receiver r, Error&& error) noexcept
            {
                ex::set_error(std::move(r.op_state.r), std::forward<Error>(error));
            }

            friend constexpr void tag_invoke(ex::set_stopped_t, receiver r) noexcept
            {
                ex::set_stopped(std::move(r.op_state.r));
            }

            // receive the MPI Request and set a callback to be
            // triggered when the mpi request completes
            constexpr void set_value(MPI_Request request) && noexcept
            {
                auto r = std::move(*this);

                // early poll just in case the request completed immediately
                if (pika::util::detail::yield_while_timeout(
                        [&]() { return !mpi::detail::poll_request(request); },
                        op_state.eager_poll_busy_wait_timeout, "trigger_mpi eager poll", false))
                {
#ifdef PIKA_HAVE_APEX
                    apex::scoped_timer apex_ invoke("pika::mpi::trigger");
#endif
                    PIKA_DETAIL_DP(mpi::detail::mpi_tran<7>,
                        debug(str<>("trigger_mpi_recv"), "eager poll ok", ptr(request)));
                    ex::set_value(std::move(r.op_state.r));
                    return;
                }

                r.op_state.request = request;

                // which polling/testing mode are we using
                mpi::detail::handler_method mode =
                    mpi::detail::get_handler_method(r.op_state.mode_flags);
                execution::thread_priority p =
                    mpi::detail::use_priority_boost(r.op_state.mode_flags) ?
                    execution::thread_priority::boost :
                    execution::thread_priority::normal;

                PIKA_DETAIL_DP(mpi::detail::mpi_tran<5>,
                    debug(str<>("trigger_mpi_recv"), "set_value_t", "req", ptr(r.op_state.request),
                        "flags", bin<8>(r.op_state.mode_flags),
                        mode_string(r.op_state.mode_flags)));

                pika::detail::try_catch_exception_ptr(
                    [&]() mutable {
                        switch (mode)
                        {
                        case mpi::detail::handler_method::yield_while:
                        {
                            // yield/while is invalid on a non pika thread
                            PIKA_ASSERT(pika::threads::detail::get_self_id());
                            pika::util::yield_while(
                                [&r]() { return !mpi::detail::poll_request(r.op_state.request); });
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

        template <typename Receiver_, typename Sender_>
        operation_state(Receiver_&& r, Sender_&& sender, int flags,
            std::chrono::duration<double> eager_poll_busy_wait_timeout)
          : r(std::forward<Receiver_>(r))
          , mode_flags{flags}
          , status{MPI_SUCCESS}
          , eager_poll_busy_wait_timeout{eager_poll_busy_wait_timeout}
          , op_state(ex::connect(std::forward<Sender_>(sender), receiver{*this}))
        {
        }

        void start() & noexcept { return ex::start(op_state); }
    };

    // -----------------------------------------------------------------
    // transform MPI adapter - sender type
    template <typename Sender>
    struct sender
    {
        PIKA_STDEXEC_SENDER_CONCEPT
        PIKA_NO_UNIQUE_ADDRESS Sender sender;
        int completion_mode_flags_;
        std::chrono::duration<double> eager_poll_busy_wait_timeout;

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
        constexpr auto connect(Receiver&& r) const&
        {
            return operation_state<std::decay_t<Receiver>, Sender>(std::forward<Receiver>(r),
                sender, completion_mode_flags_, eager_poll_busy_wait_timeout);
        }

        template <typename Receiver>
        constexpr auto connect(Receiver&& r) &&
        {
            return operation_state<std::decay_t<Receiver>, Sender>(std::forward<Receiver>(r),
                std::move(sender), completion_mode_flags_, eager_poll_busy_wait_timeout);
        }
    };
}    // namespace pika::trigger_mpi_detail

namespace pika::mpi::experimental {

    inline constexpr struct trigger_mpi_t final
      : pika::functional::detail::tag_fallback<trigger_mpi_t>
    {
    private:
        template <typename Sender,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(trigger_mpi_t, Sender&& sender, int flags,
            std::chrono::duration<double> eager_poll_busy_wait_timeout =
                std::chrono::duration<double>(0.0))
        {
            return trigger_mpi_detail::sender<std::decay_t<Sender>>{
                std::forward<Sender>(sender), flags, eager_poll_busy_wait_timeout};
        }

        //
        // tag invoke overload for mpi_trigger
        //
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(trigger_mpi_t, int flags,
            std::chrono::duration<double> eager_poll_busy_wait_timeout =
                std::chrono::duration<double>(0.0))
        {
            return pika::execution::experimental::detail::partial_algorithm<trigger_mpi_t, int,
                std::chrono::duration<double>>{flags, eager_poll_busy_wait_timeout};
        }

    } trigger_mpi{};

}    // namespace pika::mpi::experimental
