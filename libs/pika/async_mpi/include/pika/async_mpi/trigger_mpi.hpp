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
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/synchronization/condition_variable.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental::detail {
    namespace ex = pika::execution::experimental;

    // -----------------------------------------------------------------
    // route calls through an impl layer for ADL isolation
    template <typename Sender>
    struct trigger_mpi_sender_impl
    {
        struct trigger_mpi_sender_type;
    };

    template <typename Sender>
    using trigger_mpi_sender = typename trigger_mpi_sender_impl<Sender>::trigger_mpi_sender_type;

    // -----------------------------------------------------------------
    // transform MPI adapter - sender type
    template <typename Sender>
    struct trigger_mpi_sender_impl<Sender>::trigger_mpi_sender_type
    {
        PIKA_STDEXEC_SENDER_CONCEPT
        std::decay_t<Sender> sender;
        int completion_mode_flags_;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename...>
        using no_value_completion = pika::execution::experimental::completion_signatures<>;
        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<std::decay_t<Sender>,
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

        // -----------------------------------------------------------------
        // operation state for an internal receiver
        template <typename Receiver>
        struct operation_state
        {
            std::decay_t<Receiver> receiver;
            int mode_flags;
            int status;
            // these vars are needed by suspend/resume mode
            bool completed{false};
            pika::detail::spinlock mutex;
            pika::condition_variable cond_var;
            // MPI_EXT_CONTINUE
            MPI_Request request{MPI_REQUEST_NULL};

            // -----------------------------------------------------------------
            // The mpi_receiver receives inputs from the previous sender,
            // invokes the mpi call, and sets a callback on the polling handler
            struct trigger_mpi_receiver
            {
                PIKA_STDEXEC_RECEIVER_CONCEPT
                operation_state& op_state;

                template <typename Error>
                friend constexpr void
                tag_invoke(ex::set_error_t, trigger_mpi_receiver r, Error&& error) noexcept
                {
                    ex::set_error(std::move(r.op_state.receiver), std::forward<Error>(error));
                }

                friend constexpr void tag_invoke(ex::set_stopped_t, trigger_mpi_receiver r) noexcept
                {
                    ex::set_stopped(std::move(r.op_state.receiver));
                }

                // receive the MPI Request and set a callback to be
                // triggered when the mpi request completes
                constexpr void set_value(MPI_Request request) && noexcept
                {
                    auto r = std::move(*this);

                    // early exit check
                    if (request == MPI_REQUEST_NULL)
                    {
                        ex::set_value(std::move(r.op_state.receiver));
                        return;
                    }

                    r.op_state.request = request;

                    // which polling/testing mode are we using
                    handler_method mode = get_handler_method(r.op_state.mode_flags);
                    execution::thread_priority p = use_priority_boost(r.op_state.mode_flags) ?
                        execution::thread_priority::boost :
                        execution::thread_priority::normal;

                    PIKA_DETAIL_DP(mpi_tran<5>,
                        debug(str<>("trigger_mpi_recv"), "set_value_t", "req",
                            ptr(r.op_state.request), "flags", bin<8>(r.op_state.mode_flags),
                            mode_string(r.op_state.mode_flags)));

                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            switch (mode)
                            {
                            case handler_method::yield_while:
                            {
                                // yield/while is invalid on a non pika thread
                                PIKA_ASSERT(pika::threads::detail::get_self_id());
                                pika::util::yield_while(
                                    [&r]() { return !poll_request(r.op_state.request); });
#ifdef PIKA_HAVE_APEX
                                apex::scoped_timer apex_invoke("pika::mpi::trigger");
#endif
                                // we just assume the return from mpi_test is always MPI_SUCCESS
                                ex::set_value(std::move(r.op_state.receiver));
                                break;
                            }
                            case handler_method::suspend_resume:
                            {
                                // suspend is invalid on a non pika thread
                                PIKA_ASSERT(pika::threads::detail::get_self_id());
                                // the callback will resume _this_ thread
                                {
                                    std::unique_lock l{r.op_state.mutex};
                                    add_suspend_resume_request_callback(r.op_state);
                                    if (use_priority_boost(r.op_state.mode_flags))
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
                                set_value_error_helper(
                                    r.op_state.status, std::move(r.op_state.receiver));
                                break;
                            }
                            case handler_method::new_task:
                            {
                                // The callback will call set_value/set_error inside a new task
                                // and execution will continue on that thread
                                add_new_task_request_callback(r.op_state);
                                break;
                            }
                            case handler_method::continuation:
                            {
                                // The callback will call set_value/set_error
                                // execution will continue on the callback thread
                                add_continuation_request_callback(r.op_state);
                                break;
                            }
                            case handler_method::mpix_continuation:
                            {
                                PIKA_DETAIL_DP(mpi_tran<1>,
                                    debug(str<>("MPI_EXT_CONTINUE"), "register_mpix_continuation",
                                        ptr(r.op_state.request), ptr(r.op_state.request)));
                                MPIX_Continue_cb_function* func =
                                    &mpix_callback_continuation<operation_state>;
                                register_mpix_continuation(&r.op_state.request, func, &r.op_state);
                                break;
                            }
                            default: PIKA_UNREACHABLE;
                            }
                        },
                        [&](std::exception_ptr ep) {
                            ex::set_error(std::move(r.op_state.receiver), std::move(ep));
                        });
                }

                friend constexpr ex::empty_env tag_invoke(
                    ex::get_env_t, trigger_mpi_receiver const&) noexcept
                {
                    return {};
                }
            };

            using operation_state_type =
                ex::connect_result_t<std::decay_t<Sender>, trigger_mpi_receiver>;
            operation_state_type op_state;

            template <typename Receiver_, typename Sender_>
            operation_state(Receiver_&& receiver, Sender_&& sender, int flags)
              : receiver(std::forward<Receiver_>(receiver))
              , mode_flags{flags}
              , status{MPI_SUCCESS}
              , op_state(ex::connect(std::forward<Sender_>(sender), trigger_mpi_receiver{*this}))
            {
            }

            friend constexpr auto tag_invoke(ex::start_t, operation_state& os) noexcept
            {
                return ex::start(os.op_state);
            }
        };

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(ex::connect_t, trigger_mpi_sender_type const& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(std::forward<Receiver>(receiver), s.sender);
        }

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(ex::connect_t, trigger_mpi_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                std::forward<Receiver>(receiver), std::move(s.sender), s.completion_mode_flags_);
        }
    };

}    // namespace pika::mpi::experimental::detail

namespace pika::mpi::experimental {

    inline constexpr struct trigger_mpi_t final
      : pika::functional::detail::tag_fallback<trigger_mpi_t>
    {
    private:
        template <typename Sender,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(trigger_mpi_t, Sender&& sender, int flags)
        {
            return detail::trigger_mpi_sender<Sender>{std::forward<Sender>(sender), flags};
        }

        //
        // tag invoke overload for mpi_trigger
        //
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(trigger_mpi_t, int flags)
        {
            return pika::execution::experimental::detail::partial_algorithm<trigger_mpi_t, int>{
                flags};
        }

    } trigger_mpi{};

}    // namespace pika::mpi::experimental
