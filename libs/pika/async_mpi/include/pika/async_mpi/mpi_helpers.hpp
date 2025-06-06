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
#include <pika/execution/algorithms/continues_on.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/just.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/mpi_base/mpi_exception.hpp>
#include <pika/runtime/runtime.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika::mpi::experimental::detail {

    // -----------------------------------------------------------------
    // by convention the title is 7 chars (for alignment)
    template <int Level>
    inline constexpr debug::detail::print_threshold<Level, 0> mpi_tran("MPITRAN");

    // -----------------------------------------------------------------
    namespace ex = pika::execution::experimental;

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
    // return a scheduler on the default pool with added priority if requested
    inline auto default_pool_scheduler(execution::thread_priority p)
    {
        return ex::with_priority(
            ex::thread_pool_scheduler{
                &pika::detail::get_runtime_ptr()->get_thread_manager().default_pool()},
            p);
    }

    // -----------------------------------------------------------------
    // return a scheduler on the mpi pool
    inline auto mpi_pool_scheduler(execution::thread_priority p)
    {
        if (!get_pool_enabled()) return default_pool_scheduler(p);
        return ex::with_priority(
            ex::thread_pool_scheduler{&resource::get_thread_pool(get_pool_name())}, p);
    }

    // -----------------------------------------------------------------
    // depending on mpi_status : calls set_value (with Ts...) or set_error on the receiver
    template <typename Receiver, typename... Ts>
    void set_value_error_helper(int mpi_status, Receiver&& receiver, Ts&&... ts)
    {
        static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");
        if (mpi_status == MPI_SUCCESS)
        {
            ex::set_value(std::forward<Receiver>(receiver), std::forward<Ts>(ts)...);
        }
        else
        {
            ex::set_error(std::forward<Receiver>(receiver),
                std::make_exception_ptr(mpi::exception(mpi_status, "set_error handler")));
        }
    }

    // -----------------------------------------------------------------
    // handler_method::suspend_resume
    // adds a request callback to the mpi polling code which will call
    // notify_one to wake up a suspended task
    template <typename OperationState>
    void add_suspend_resume_request_callback(OperationState& op_state)
    {
        PIKA_ASSERT(op_state.completed == false);
        detail::add_request_callback(
            [&op_state](int status) mutable {
                PIKA_DETAIL_DP(
                    mpi_tran<5>, debug(str<>("callback_void_suspend_resume"), "status", status));
                op_state.ts = {};
                // wake up the suspended thread
                {
                    std::lock_guard lk(op_state.mutex);
                    op_state.status = status;
                    op_state.completed = true;
                }
                op_state.cond_var.notify_one();
            },
            op_state.request);
    }

    // -----------------------------------------------------------------
    // handler_method::new_task
    // adds a request callback to the mpi polling code which will call
    // set_value/error on the receiver
    template <typename OperationState>
    void add_new_task_request_callback(OperationState& op_state)
    {
        detail::add_request_callback(
            [&op_state](int status) mutable {
                PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("schedule_task_callback")));
                op_state.ts = {};
                if (status != MPI_SUCCESS)
                {
                    ex::set_error(std::move(op_state.r),
                        std::make_exception_ptr(
                            mpi::exception(status, "new_task_request_callback")));
                }
                else
                {
                    // pass the result onto a new task and invoke the continuation
                    execution::thread_priority p = use_priority_boost(op_state.mode_flags) ?
                        execution::thread_priority::boost :
                        execution::thread_priority::normal;
                    auto snd0 =
                        ex::schedule(default_pool_scheduler(p)) | ex::then([&op_state]() mutable {
                            PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("set_value")));
                            ex::set_value(std::move(op_state.r));
                        });
                    ex::start_detached(std::move(snd0));
                }
            },
            op_state.request);
    }

    // -----------------------------------------------------------------
    // handler_method::continuation
    // adds a request callback to the mpi polling code which will call
    // the set_value/set_error helper using the void return signature
    template <typename OperationState>
    void add_continuation_request_callback(OperationState& op_state)
    {
        detail::add_request_callback(
            [&op_state](int status) mutable {
                PIKA_DETAIL_DP(mpi_tran<5>, debug(str<>("callback_void")));
                op_state.ts = {};
                set_value_error_helper(status, std::move(op_state.r));
            },
            op_state.request);
    }

    // -----------------------------------------------------------------
    // handler_method::mpix_continuation - signature is
    /// typedef int (MPIX_Continue_cb_function)(int rc, void *cb_data);
    template <typename OperationState>
    int mpix_callback_resume([[maybe_unused]] int rc, void* cb_data)
    {
        PIKA_DETAIL_DP(mpi_tran<1>, debug(str<>("MPIX"), "callback triggered"));
        auto& op_state = *static_cast<OperationState*>(cb_data);
        // wake up the suspended thread
        {
            std::lock_guard lk(op_state.mutex);
            op_state.status = rc;
            op_state.completed = true;
        }
        op_state.cond_var.notify_one();
        // tell mpix that we handled it ok, error is passed into set_error in mpi_trigger
        return MPI_SUCCESS;
    }

    // -----------------------------------------------------------------
    // handler_method::mpix_continuation2 - signature is
    /// typedef int (MPIX_Continue_cb_function)(int rc, void *cb_data);
    template <typename OperationState>
    int mpix_callback_continuation([[maybe_unused]] int rc, void* cb_data)
    {
        PIKA_DETAIL_DP(mpi_tran<1>, debug(str<>("MPIX"), "callback triggered"));
        auto& op_state = *static_cast<OperationState*>(cb_data);
        set_value_error_helper(op_state.status, std::move(op_state.r));
        // tell mpix that we handled it ok, error is passed into set_error in mpi_trigger
        return MPI_SUCCESS;
    }

}    // namespace pika::mpi::experimental::detail
