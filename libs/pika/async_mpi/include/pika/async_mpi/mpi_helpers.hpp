//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_xxx.hpp

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
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/execution_base/any_sender.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/executors/inline_scheduler.hpp>
#include <pika/executors/limiting_scheduler.hpp>
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

    // -----------------------------------------------------------------
    // by convention the title is 7 chars (for alignment)
    using print_on = pika::debug::detail::enable_print<false>;
    inline constexpr print_on mpi_tran("MPITRAN");

    namespace pud = pika::util::detail;
    namespace exp = execution::experimental;

    // -----------------------------------------------------------------
    template <typename T>
    struct any_sender_helper
    {
        using type = exp::unique_any_sender<T>;
    };

    template <>
    struct any_sender_helper<void>
    {
        using type = exp::unique_any_sender<>;
    };

    // -----------------------------------------------------------------
    // is func(Ts..., MPI_Request) invokable
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
    inline auto mpi_pool_scheduler(bool stack = true)
    {
        if (stack)
        {
            return exp::thread_pool_scheduler{&resource::get_thread_pool(get_pool_name())};
        }
        else
        {
            return exp::with_stacksize(
                exp::thread_pool_scheduler{&resource::get_thread_pool(get_pool_name())},
                execution::thread_stacksize::nostack);
        }
    }

    // -----------------------------------------------------------------
    // return a scheduler on the default pool with added priority if requested
    inline auto default_pool_scheduler(
        execution::thread_priority p = execution::thread_priority::normal)
    {
        if (p == execution::thread_priority::normal)
        {
            return exp::thread_pool_scheduler{&resource::get_thread_pool("default")};
        }
        return exp::with_priority(
            exp::thread_pool_scheduler{&resource::get_thread_pool("default")}, p);
    }

    // -----------------------------------------------------------------
    // depending on mpi_status : calls set_value (with Ts...) or set_error on the receiver
    template <typename Receiver, typename... Ts>
    void set_value_error_helper(int mpi_status, Receiver&& receiver, Ts&&... ts)
    {
        static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");
        if (mpi_status == MPI_SUCCESS)
        {
            exp::set_value(PIKA_FORWARD(Receiver, receiver), PIKA_FORWARD(Ts, ts)...);
        }
        else
        {
            exp::set_error(PIKA_FORWARD(Receiver, receiver),
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
                using namespace pika::debug::detail;
                PIKA_DETAIL_DP(mpi_tran,
                    debug(str<>("callback_void"), "stream", detail::stream_name(op_state.stream)));
                op_state.ts = {};
                set_value_error_helper(status, PIKA_MOVE(op_state.receiver));
            },
            request, detail::check_request_eager::yes);
    }

    // -----------------------------------------------------------------
    // adds a request callback to the mpi polling code which will call
    // the set_value/set_error helper with a valid return value
    template <typename Result, typename OperationState>
    void set_value_request_callback_non_void(MPI_Request request, OperationState& op_state)
    {
        detail::add_request_callback(
            [&op_state](int status) mutable {
                using namespace pika::debug::detail;
                PIKA_DETAIL_DP(mpi_tran,
                    debug(
                        str<>("callback_nonvoid"), "stream", detail::stream_name(op_state.stream)));
                op_state.ts = {};
                PIKA_ASSERT(std::holds_alternative<Result>(op_state.result));
                set_value_error_helper(status, PIKA_MOVE(op_state.receiver),
                    PIKA_MOVE(std::get<Result>(op_state.result)));
            },
            request, detail::check_request_eager::yes);
    }

    // -----------------------------------------------------------------
    // adds a request callback to the mpi polling code which will call
    // notify_one to wake up a suspended task
    template <typename OperationState>
    void resume_request_callback(MPI_Request request, OperationState& op_state)
    {
        detail::add_request_callback(
            [&op_state](int status) mutable {
                using namespace pika::debug::detail;
                PIKA_DETAIL_DP(mpi_tran,
                    debug(str<>("callback_void_suspend_resume"), "stream",
                        detail::stream_name(op_state.stream)));
                op_state.ts = {};
                op_state.status = status;

                // wake up the suspended thread
                {
                    std::lock_guard lk(op_state.mutex_);
                    op_state.completed = true;
                }
                op_state.cond_var_.notify_one();
            },
            // we do not need to eagerly check, because it was done earlier
            request, detail::check_request_eager::no);
    }

}    // namespace pika::mpi::experimental::detail
