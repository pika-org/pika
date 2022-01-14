//  Copyright (c) 2019-2020 ETH Zurich
//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/execution/detail/async_launch_policy_dispatch.hpp>
#include <pika/execution/detail/post_policy_dispatch.hpp>
#include <pika/execution/detail/sync_launch_policy_dispatch.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/execution/executors/fused_bulk_execute.hpp>
#include <pika/execution/executors/static_chunk_size.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/executors/detail/hierarchical_spawning.hpp>
#include <pika/functional/bind_back.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/one_shot.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/serialization/serialize.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { namespace execution { namespace detail {
    template <typename Policy>
    struct get_default_policy
    {
        static constexpr Policy call() noexcept
        {
            return Policy{};
        }
    };

    template <>
    struct get_default_policy<pika::launch>
    {
        static constexpr pika::launch::async_policy call() noexcept
        {
            return pika::launch::async_policy{};
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename... Ts>
    struct bulk_function_result;

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result;

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct then_bulk_function_result;
}}}}    // namespace pika::parallel::execution::detail

namespace pika { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// A \a parallel_executor creates groups of parallel execution agents
    /// which execute in threads implicitly created by the executor. This
    /// executor prefers continuing with the creating thread first before
    /// executing newly created threads.
    ///
    /// This executor conforms to the concepts of a TwoWayExecutor,
    /// and a BulkTwoWayExecutor
    template <typename Policy>
    struct parallel_policy_executor
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        using execution_category = parallel_execution_tag;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        using executor_parameters_type = static_chunk_size;

        /// Create a new parallel executor
        constexpr explicit parallel_policy_executor(
            threads::thread_priority priority,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            threads::thread_schedule_hint schedulehint = {},
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call(),
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : pool_(nullptr)
          , policy_(l, priority, stacksize, schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint schedulehint = {},
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call())
          : pool_(nullptr)
          , policy_(
                l, threads::thread_priority::default_, stacksize, schedulehint)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_schedule_hint schedulehint,
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call())
          : pool_(nullptr)
          , policy_(l, threads::thread_priority::default_,
                threads::thread_stacksize::default_, schedulehint)
        {
        }

        constexpr explicit parallel_policy_executor(
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call())
          : pool_(nullptr)
          , policy_(l)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_pool_base* pool,
            threads::thread_priority priority =
                threads::thread_priority::default_,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            threads::thread_schedule_hint schedulehint = {},
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call(),
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : pool_(pool)
          , policy_(l, priority, stacksize, schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
        {
        }

        // property implementations
        friend constexpr parallel_policy_executor tag_invoke(
            pika::execution::experimental::with_hint_t,
            parallel_policy_executor const& exec,
            pika::threads::thread_schedule_hint hint)
        {
            auto exec_with_hint = exec;
            exec_with_hint.policy_ = hint;
            return exec_with_hint;
        }

        friend constexpr pika::threads::thread_schedule_hint tag_invoke(
            pika::execution::experimental::get_hint_t,
            parallel_policy_executor const& exec) noexcept
        {
            return exec.policy_.hint();
        }

        friend constexpr parallel_policy_executor tag_invoke(
            pika::execution::experimental::with_priority_t,
            parallel_policy_executor const& exec,
            pika::threads::thread_priority priority)
        {
            auto exec_with_priority = exec;
            exec_with_priority.policy_ = priority;
            return exec_with_priority;
        }

        friend constexpr pika::threads::thread_priority tag_invoke(
            pika::execution::experimental::get_priority_t,
            parallel_policy_executor const& exec) noexcept
        {
            return exec.policy_.priority();
        }

        friend constexpr parallel_policy_executor tag_invoke(
            pika::execution::experimental::with_annotation_t,
            parallel_policy_executor const& exec, char const* annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ = annotation;
            return exec_with_annotation;
        }

        friend parallel_policy_executor tag_invoke(
            pika::execution::experimental::with_annotation_t,
            parallel_policy_executor const& exec, std::string annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ =
                detail::store_function_annotation(PIKA_MOVE(annotation));
            return exec_with_annotation;
        }

        friend constexpr char const* tag_invoke(
            pika::execution::experimental::get_annotation_t,
            parallel_policy_executor const& exec) noexcept
        {
            return exec.annotation_;
        }

        /// \cond NOINTERNAL
        constexpr bool operator==(
            parallel_policy_executor const& rhs) const noexcept
        {
            return policy_ == rhs.policy_ && pool_ == rhs.pool_ &&
                hierarchical_threshold_ == rhs.hierarchical_threshold_;
        }

        constexpr bool operator!=(
            parallel_policy_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        constexpr parallel_policy_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        typename pika::util::detail::invoke_deferred_result<F, Ts...>::type
        sync_execute(F&& f, Ts&&... ts) const
        {
            pika::scoped_annotation annotate(annotation_ ?
                    annotation_ :
                    "parallel_policy_executor::sync_execute");
            return pika::detail::sync_launch_policy_dispatch<Policy>::call(
                launch::sync, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        pika::future<
            typename pika::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            pika::util::thread_description desc(f, annotation_);
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();

            return pika::detail::async_launch_policy_dispatch<Policy>::call(
                policy_, desc, pool, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        PIKA_FORCEINLINE
            pika::future<typename pika::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type>
            then_execute(F&& f, Future&& predecessor, Ts&&... ts) const
        {
            using result_type =
                typename pika::util::detail::invoke_deferred_result<F, Future,
                    Ts...>::type;

            auto&& func = pika::util::one_shot(pika::util::bind_back(
                pika::annotated_function(PIKA_FORWARD(F, f), annotation_),
                PIKA_FORWARD(Ts, ts)...));

            typename pika::traits::detail::shared_state_ptr<result_type>::type
                p = lcos::detail::make_continuation_alloc_nounwrap<result_type>(
                    pika::util::internal_allocator<>{},
                    PIKA_FORWARD(Future, predecessor), policy_, PIKA_MOVE(func));

            return pika::traits::future_access<pika::future<result_type>>::create(
                PIKA_MOVE(p));
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            pika::util::thread_description desc(f, annotation_);
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();
            parallel::execution::detail::post_policy_dispatch<Policy>::call(
                policy_, desc, pool, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<pika::future<typename parallel::execution::detail::
                bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            pika::util::thread_description desc(f, annotation_);
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();
            return parallel::execution::detail::
                hierarchical_bulk_async_execute_helper(desc, pool, 0,
                    pool->get_os_thread_count(), hierarchical_threshold_,
                    policy_, PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        pika::future<typename parallel::execution::detail::
                bulk_then_execute_result<F, S, Future, Ts...>::type>
        bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            return parallel::execution::detail::
                hierarchical_bulk_then_execute_helper(*this, policy_,
                    pika::annotated_function(PIKA_FORWARD(F, f), annotation_),
                    shape, PIKA_FORWARD(Future, predecessor),
                    PIKA_FORWARD(Ts, ts)...);
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class pika::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & policy_ & hierarchical_threshold_;
            // clang-format on
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        static constexpr std::size_t hierarchical_threshold_default_ = 6;

        threads::thread_pool_base* pool_;
        Policy policy_;
        std::size_t hierarchical_threshold_ = hierarchical_threshold_default_;
        char const* annotation_ = nullptr;
        /// \endcond
    };

    using parallel_executor = parallel_policy_executor<pika::launch>;
}}    // namespace pika::execution

namespace pika { namespace parallel { namespace execution {
    using parallel_executor PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::execution::parallel_executor is deprecated. Use "
        "pika::execution::parallel_executor instead.") =
        pika::execution::parallel_executor;
    template <typename Policy>
    using parallel_policy_executor PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::execution::parallel_policy_executor is deprecated. Use "
        "pika::execution::parallel_policy_executor instead.") =
        pika::execution::parallel_policy_executor<Policy>;
}}}    // namespace pika::parallel::execution

namespace pika { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <typename Policy>
    struct is_one_way_executor<pika::execution::parallel_policy_executor<Policy>>
      : std::true_type
    {
    };

    template <typename Policy>
    struct is_two_way_executor<pika::execution::parallel_policy_executor<Policy>>
      : std::true_type
    {
    };

    template <typename Policy>
    struct is_bulk_two_way_executor<
        pika::execution::parallel_policy_executor<Policy>> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace pika::parallel::execution
