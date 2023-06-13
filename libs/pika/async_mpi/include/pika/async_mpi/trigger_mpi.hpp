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
#include <pika/async_mpi/mpi_exception.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/debugging/demangle_helper.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental::detail {

    namespace pud = pika::util::detail;

    // -----------------------------------------------------------------
    // route calls through an impl layer for ADL resolution
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
        using is_sender = void;
        std::decay_t<Sender> sender;
        int completion_mode_flags_;

        // -----------------------------------------------------------------
        // completion signatures
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <typename...> class Variant>
        using error_types = pud::unique_t<
            pud::prepend_t<typename exp::sender_traits<Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;

        // -----------------------------------------------------------------
        // operation state for a given receiver
        template <typename Receiver>
        struct operation_state
        {
            std::decay_t<Receiver> receiver;
            int mode_flags_;
            int status;
            // these vars are needed by suspend/resume mode
            bool completed;
            pika::detail::spinlock mutex_;
            pika::condition_variable cond_var_;

            // -----------------------------------------------------------------
            // The mpi_receiver receives inputs from the previous sender,
            // invokes the mpi call, and sets a callback on the polling handler
            struct trigger_mpi_receiver
            {
                using is_receiver = void;
                operation_state& op_state;

                template <typename Error>
                friend constexpr void
                tag_invoke(exp::set_error_t, trigger_mpi_receiver&& r, Error&& error) noexcept
                {
                    exp::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_FORWARD(Error, error));
                }

                friend constexpr void tag_invoke(
                    exp::set_stopped_t, trigger_mpi_receiver&& r) noexcept
                {
                    exp::set_stopped(PIKA_MOVE(r.op_state.receiver));
                }

                // receive the MPI Request and set a callback to be
                // triggered when the mpi request completes
                friend constexpr void tag_invoke(
                    exp::set_value_t, trigger_mpi_receiver&& r, MPI_Request request) noexcept
                {
                    // early exit check
                    if (request == MPI_REQUEST_NULL)
                    {
                        exp::set_value(PIKA_MOVE(r.op_state.receiver));
                        return;
                    }

                    // which polling/testing mode are we using
                    handler_mode mode = get_handler_mode(r.op_state.mode_flags_);

                    PIKA_DETAIL_DP(mpi_tran<5>,
                        debug(str<>("trigger_mpi_recv"), "set_value_t", "req", ptr(request),
                            "flags", bin<8>(r.op_state.mode_flags_),
                            mode_string(r.op_state.mode_flags_)));

                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            switch (mode)
                            {
                            case handler_mode::yield_while:
                            {
                                pika::util::yield_while(
                                    [&request]() { return !detail::poll_request(request); });
                                // we just assume the return from mpi_test is always MPI_SUCCESS
                                exp::set_value(PIKA_MOVE(r.op_state.receiver));
                                break;
                            }
                            case handler_mode::new_task:
                            {
                                // The callback will call set_value/set_error inside a new task
                                // and execution will continue on that thread
                                detail::schedule_task_callback(
                                    request, PIKA_MOVE(r.op_state.receiver));
                                break;
                            }
                            case handler_mode::continuation:
                            {
                                // The callback will call set_value/set_error
                                // execution will continue on the callback thread
                                detail::set_value_request_callback_void<>(request, r.op_state);
                                break;
                            }
                            case handler_mode::suspend_resume:
                            {
                                // suspend is invalid except on a pika thread
                                PIKA_ASSERT(pika::threads::detail::get_self_id());
                                // the callback will resume _this_ thread
                                std::unique_lock l{r.op_state.mutex_};
                                resume_request_callback(request, r.op_state);
                                if (use_HP_com(r.op_state.mode_flags_))
                                {
                                    threads::detail::thread_data::scoped_thread_priority
                                        set_restore(execution::thread_priority::high);
                                    r.op_state.cond_var_.wait(
                                        l, [&]() { return r.op_state.completed; });
                                }
                                else
                                {
                                    r.op_state.cond_var_.wait(
                                        l, [&]() { return r.op_state.completed; });
                                }
                                // call set_value/set_error depending on mpi return status
                                set_value_error_helper(
                                    r.op_state.status, PIKA_MOVE(r.op_state.receiver));
                                break;
                            }
                            default:
                            {
                                // I bet the "review" police don't like this exception
                                throw mpi_exception(MPI_ERR_UNKNOWN, "eek!");
                                break;
                            }
                            }
                        },
                        [&](std::exception_ptr ep) {
                            exp::set_error(PIKA_MOVE(r.op_state.receiver), PIKA_MOVE(ep));
                        });
                }

                friend constexpr exp::empty_env tag_invoke(
                    exp::get_env_t, trigger_mpi_receiver const&) noexcept
                {
                    return {};
                }
            };

            using operation_state_type =
                exp::connect_result_t<std::decay_t<Sender>, trigger_mpi_receiver>;
            operation_state_type op_state;

            template <typename Receiver_, typename Sender_>
            operation_state(Receiver_&& receiver, Sender_&& sender, int flags)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , mode_flags_{flags}
              , op_state(exp::connect(PIKA_FORWARD(Sender_, sender), trigger_mpi_receiver{*this}))
            {
            }

            friend constexpr auto tag_invoke(exp::start_t, operation_state& os) noexcept
            {
                return exp::start(os.op_state);
            }
        };

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(exp::connect_t, trigger_mpi_sender_type const& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(PIKA_FORWARD(Receiver, receiver), s.sender);
        }

        template <typename Receiver>
        friend constexpr auto
        tag_invoke(exp::connect_t, trigger_mpi_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.sender), s.completion_mode_flags_);
        }
    };

}    // namespace pika::mpi::experimental::detail

namespace pika::mpi::experimental {

    namespace exp = pika::execution::experimental;

    inline constexpr struct trigger_mpi_t final
      : pika::functional::detail::tag_fallback<trigger_mpi_t>
    {
    private:
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(exp::is_sender_v<std::decay_t<Sender>>)>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(trigger_mpi_t, Sender&& sender, int flags)
        {
            return detail::trigger_mpi_sender<Sender>{PIKA_FORWARD(Sender, sender), flags};
        }

        //
        // tag invoke overload for mpi_trigger
        //
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(trigger_mpi_t, int flags)
        {
            return exp::detail::partial_algorithm<trigger_mpi_t, int>{flags};
        }

    } trigger_mpi{};

}    // namespace pika::mpi::experimental
