//  Copyright (c)      2020 Mikael Simberg
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/restricted_thread_pool_executors.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/executors/thread_pool_executor.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika::parallel::execution {
    class restricted_thread_pool_executor
    {
        static constexpr std::size_t hierarchical_threshold_default_ = 6;

    public:
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        using execution_category = pika::execution::parallel_execution_tag;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        using executor_parameters_type = pika::execution::static_chunk_size;

        /// Create a new parallel executor
        restricted_thread_pool_executor(std::size_t first_thread = 0, std::size_t num_threads = 1,
            pika::execution::thread_priority priority = pika::execution::thread_priority::default_,
            pika::execution::thread_stacksize stacksize =
                pika::execution::thread_stacksize::default_,
            pika::execution::thread_schedule_hint schedulehint = {},
            std::size_t hierarchical_threshold = hierarchical_threshold_default_)
          : pool_(this_thread::get_pool())
          , priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
          , first_thread_(first_thread)
          , num_threads_(num_threads)
          , os_thread_(first_thread_)
        {
            PIKA_ASSERT(pool_);
        }

        restricted_thread_pool_executor(restricted_thread_pool_executor const& other)
          : pool_(other.pool_)
          , priority_(other.priority_)
          , stacksize_(other.stacksize_)
          , schedulehint_(other.schedulehint_)
          , first_thread_(other.first_thread_)
          , num_threads_(other.num_threads_)
          , os_thread_(other.first_thread_)
        {
            PIKA_ASSERT(pool_);
        }

        /// \cond NOINTERNAL
        bool operator==(restricted_thread_pool_executor const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ && schedulehint_ == rhs.schedulehint_ &&
                first_thread_ == rhs.first_thread_ && num_threads_ == rhs.num_threads_;
        }

        bool operator!=(restricted_thread_pool_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        restricted_thread_pool_executor const& context() const noexcept
        {
            return *this;
        }

    private:
        std::uint16_t get_next_thread_num()
        {
            return static_cast<std::uint16_t>(first_thread_ + (os_thread_++ % num_threads_));
        }

    public:
        template <typename F, typename... Ts>
        pika::future<typename pika::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts)
        {
            pika::detail::thread_description desc(f);

            auto policy = launch::async_policy(priority_, stacksize_,
                pika::execution::thread_schedule_hint(get_next_thread_num()));

            return pika::detail::async_launch_policy_dispatch<launch::async_policy>::call(
                policy, desc, pool_, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        pika::future<typename pika::util::detail::invoke_deferred_result<F, Future, Ts...>::type>
        then_execute(F&& f, Future&& predecessor, Ts&&... ts)
        {
            using result_type =
                typename pika::util::detail::invoke_deferred_result<F, Future, Ts...>::type;

            auto&& func = pika::util::detail::one_shot(
                pika::util::detail::bind_back(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...));

            typename pika::traits::detail::shared_state_ptr<result_type>::type p =
                pika::lcos::detail::make_continuation_exec<result_type>(
                    PIKA_FORWARD(Future, predecessor), *this, PIKA_MOVE(func));

            return pika::traits::future_access<pika::future<result_type>>::create(PIKA_MOVE(p));
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            pika::detail::thread_description desc(f);

            auto policy = launch::async_policy(priority_, stacksize_,
                pika::execution::thread_schedule_hint(get_next_thread_num()));

            detail::post_policy_dispatch<launch::async_policy>::call(
                policy, desc, pool_, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename... Ts>
        std::vector<pika::future<typename detail::bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            auto policy = launch::async_policy(priority_, stacksize_, schedulehint_);

            return detail::hierarchical_bulk_async_execute_helper(pool_, first_thread_,
                num_threads_, hierarchical_threshold_, policy, PIKA_FORWARD(F, f), shape,
                PIKA_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        pika::future<typename detail::bulk_then_execute_result<F, S, Future, Ts...>::type>
        bulk_then_execute(F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            return detail::hierarchical_bulk_then_execute_helper(*this, launch::async,
                PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Future, predecessor),
                PIKA_FORWARD(Ts, ts)...);
        }
        /// \endcond

    private:
        threads::detail::thread_pool_base* pool_ = nullptr;

        pika::execution::thread_priority priority_ = pika::execution::thread_priority::default_;
        pika::execution::thread_stacksize stacksize_ = pika::execution::thread_stacksize::default_;
        pika::execution::thread_schedule_hint schedulehint_ = {};
        std::size_t hierarchical_threshold_ = hierarchical_threshold_default_;

        std::size_t first_thread_;
        std::size_t num_threads_;
        std::atomic<std::size_t> os_thread_;
    };
}    // namespace pika::parallel::execution

namespace pika::parallel::execution {
    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<parallel::execution::restricted_thread_pool_executor>
      : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<parallel::execution::restricted_thread_pool_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<parallel::execution::restricted_thread_pool_executor>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace pika::parallel::execution
