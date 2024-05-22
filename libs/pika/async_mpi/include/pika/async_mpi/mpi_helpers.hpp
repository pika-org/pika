//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/debugging/demangle_helper.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/just.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental::detail {

    // -----------------------------------------------------------------
    // by convention the title is 7 chars (for alignment)
    template <int Level>
    inline constexpr debug::detail::print_threshold<Level, 0> mpi_tran("MPITRAN");

    namespace ex = pika::execution::experimental;

    // -----------------------------------------------------------------
    template <typename T>
    struct any_sender_helper
    {
        using type = ex::unique_any_sender<T>;
    };

    template <>
    struct any_sender_helper<void>
    {
        using type = ex::unique_any_sender<>;
    };

    // -----------------------------------------------------------------
    // is func(Ts..., MPI_Request) invocable
    template <typename F, typename... Ts>
    inline constexpr bool is_mpi_request_invocable_v =
        std::is_invocable_v<F, std::add_lvalue_reference_t<std::decay_t<Ts>>..., MPI_Request*>;

    // -----------------------------------------------------------------
    // get return type of func(Ts..., MPI_Request)
    template <typename F, typename... Ts>
    using mpi_request_invoke_result_t = std::decay_t<
        std::invoke_result_t<F, std::add_lvalue_reference_t<std::decay_t<Ts>>..., MPI_Request*>>;

    // -----------------------------------------------------------------
    // return a scheduler on the mpi pool, with or without stack
    inline auto mpi_pool_scheduler(execution::thread_priority p, bool stack = true)
    {
        ex::thread_pool_scheduler sched{&resource::get_thread_pool(get_pool_name())};
        if (!stack)
        {
            sched = ex::with_stacksize(std::move(sched), execution::thread_stacksize::nostack);
        }
        sched = ex::with_priority(std::move(sched), p);
        return sched;
    }

    // -----------------------------------------------------------------
    // return a scheduler on the default pool with added priority if requested
    inline auto default_pool_scheduler(execution::thread_priority p)
    {
        return ex::with_priority(
            ex::thread_pool_scheduler{&resource::get_thread_pool("default")}, p);
    }

    // -----------------------------------------------------------------
    // depending on mpi_status : calls set_value (with Ts...) or set_error on the receiver
    template <typename Receiver, typename... Ts>
    void set_value_error_helper(int mpi_status, Receiver&& receiver, Ts&&... ts)
    {
        static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");
        if (mpi_status == MPI_SUCCESS)
        {
            ex::set_value(PIKA_FORWARD(Receiver, receiver), PIKA_FORWARD(Ts, ts)...);
        }
        else
        {
            ex::set_error(PIKA_FORWARD(Receiver, receiver),
                std::make_exception_ptr(mpi_exception(mpi_status)));
        }
    }

    // -----------------------------------------------------------------
    // adds a request callback to the mpi polling code which will call
    // the set_value/set_error helper using the void return signature
    template <typename OperationState>
    void set_value_request_callback_void(MPI_Request request, OperationState& op_state)
    {
        detail::add_request_callback(
            [&op_state](int status) mutable {
                PIKA_DETAIL_DP(mpi_tran<5>,
                    debug(str<>(
                        "callback_void") /*, "stream", detail::stream_name(op_state.stream)*/));
                set_value_error_helper(status, PIKA_MOVE(op_state.receiver));
            },
            request);
    }

    // -----------------------------------------------------------------
    // adds a request callback to the mpi polling code which will call
    // the set_value/set_error helper with a valid return value
    template <typename Result, typename OperationState>
    void set_value_request_callback_non_void(MPI_Request request, OperationState& op_state)
    {
        detail::add_request_callback(
            [&op_state](int status) mutable {
                PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("callback_nonvoid")));
                PIKA_ASSERT(std::holds_alternative<Result>(op_state.result));
                set_value_error_helper(status, PIKA_MOVE(op_state.receiver),
                    PIKA_MOVE(std::get<Result>(op_state.result)));
            },
            request);
    }

    // -----------------------------------------------------------------
    // adds a request callback to the mpi polling code which will call
    // notify_one to wake up a suspended task
    template <typename OperationState>
    void resume_request_callback(MPI_Request request, OperationState& op_state)
    {
        PIKA_ASSERT(op_state.completed == false);
        detail::add_request_callback(
            [&op_state](int status) mutable {
                PIKA_DETAIL_DP(mpi_tran<5>,
                    debug(str<>("callback_void_suspend_resume"), "status", status
                        /*, "stream", detail::stream_name(op_state.stream)*/));
                // wake up the suspended thread
                {
                    std::lock_guard lk(op_state.mutex);
                    op_state.status = status;
                    op_state.completed = true;
                }
                op_state.cond_var.notify_one();
            },
            request);
    }

    // -----------------------------------------------------------------
    /// typedef int (MPIX_Continue_cb_function)(int rc, void *cb_data);
    template <typename OperationState>
    static int mpix_callback([[maybe_unused]] int rc, void* cb_data)
    {
        PIKA_DETAIL_DP(mpi_tran<1>, debug(str<>("MPIX"), "callback triggered"));
        auto& op_state = *static_cast<OperationState*>(cb_data);
        // wake up the suspended thread
        {
            std::lock_guard lk(op_state.mutex);
            op_state.completed = true;
        }
        op_state.cond_var.notify_one();
        return MPI_SUCCESS;
    }

    // -----------------------------------------------------------------
    // adds a request callback to the mpi polling code which will call
    // set_value/error on the receiver
    template <typename Receiver>
    void
    schedule_task_callback(MPI_Request request, execution::thread_priority p, Receiver&& receiver)
    {
        detail::add_request_callback(
            [receiver = PIKA_MOVE(receiver), p](int status) mutable {
                PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("schedule_task_callback")));
                if (status != MPI_SUCCESS)
                {
                    ex::set_error(PIKA_FORWARD(Receiver, receiver),
                        std::make_exception_ptr(mpi_exception(status)));
                }
                else
                {
                    // pass the result onto a new task and invoke the continuation
                    auto snd0 = ex::just(status) | ex::transfer(default_pool_scheduler(p)) |
                        ex::then([receiver = PIKA_MOVE(receiver)](int status) mutable {
                            PIKA_DETAIL_DP(
                                mpi_tran<5>, debug(str<>("set_value_error_helper"), status));
                            set_value_error_helper(status, PIKA_MOVE(receiver));
                        });
                    ex::start_detached(PIKA_MOVE(snd0));
                }
            },
            request);
    }

}    // namespace pika::mpi::experimental::detail
