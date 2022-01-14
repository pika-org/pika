//  Copyright (c) 2019-2020 ETH Zurich
//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/execution/detail/async_launch_policy_dispatch.hpp>
#include <pika/execution/detail/post_policy_dispatch.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/fused_bulk_execute.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/pack_traversal/unwrap.hpp>
#include <pika/synchronization/latch.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { namespace execution { namespace detail {

    template <typename Launch, typename F, typename S, typename... Ts>
    std::vector<
        pika::future<typename detail::bulk_function_result<F, S, Ts...>::type>>
    hierarchical_bulk_async_execute_helper(
        pika::util::thread_description const& desc,
        threads::thread_pool_base* pool, std::size_t first_thread,
        std::size_t num_threads, std::size_t hierarchical_threshold,
        Launch policy, F&& f, S const& shape, Ts&&... ts)
    {
        PIKA_ASSERT(pool);

        typedef std::vector<pika::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
            result_type;

        result_type results;
        std::size_t const size = pika::util::size(shape);
        results.resize(size);

        auto post_policy = policy;
        post_policy.set_stacksize(threads::thread_stacksize::small_);

        lcos::local::latch l(size);
        std::size_t part_begin = 0;
        auto it = std::begin(shape);
        for (std::size_t t = 0; t < num_threads; ++t)
        {
            std::size_t const part_end = ((t + 1) * size) / num_threads;
            std::size_t const part_size = part_end - part_begin;

            auto async_policy = policy;
            async_policy.set_hint(threads::thread_schedule_hint{
                static_cast<std::int16_t>(first_thread + t)});

            if (part_size > hierarchical_threshold)
            {
                detail::post_policy_dispatch<Launch>::call(post_policy, desc,
                    pool,
                    [&, part_begin, part_end, part_size, f, it]() mutable {
                        for (std::size_t part_i = part_begin; part_i < part_end;
                             ++part_i)
                        {
                            results[part_i] =
                                pika::detail::async_launch_policy_dispatch<
                                    Launch>::call(async_policy, desc, pool, f,
                                    *it, ts...);
                            ++it;
                        }
                        l.count_down(part_size);
                    });

                std::advance(it, part_size);
            }
            else
            {
                for (std::size_t part_i = part_begin; part_i < part_end;
                     ++part_i)
                {
                    results[part_i] =
                        pika::detail::async_launch_policy_dispatch<Launch>::call(
                            async_policy, desc, pool, f, *it, ts...);
                    ++it;
                }
                l.count_down(part_size);
            }

            part_begin = part_end;
        }

        l.wait();

        return results;
    }

    template <typename Launch, typename F, typename S, typename... Ts>
    std::vector<
        pika::future<typename detail::bulk_function_result<F, S, Ts...>::type>>
    hierarchical_bulk_async_execute_helper(threads::thread_pool_base* pool,
        std::size_t first_thread, std::size_t num_threads,
        std::size_t hierarchical_threshold, Launch policy, F&& f,
        S const& shape, Ts&&... ts)
    {
        pika::util::thread_description const desc(f,
            "pika::parallel::execution::detail::hierarchical_bulk_async_execute_"
            "helper");

        return hierarchical_bulk_async_execute_helper(desc, pool, first_thread,
            num_threads, hierarchical_threshold, policy, PIKA_FORWARD(F, f),
            shape, PIKA_FORWARD(Ts, ts)...);
    }

    template <typename Executor, typename Launch, typename F, typename S,
        typename Future, typename... Ts>
    pika::future<
        typename detail::bulk_then_execute_result<F, S, Future, Ts...>::type>
    hierarchical_bulk_then_execute_helper(Executor&& executor, Launch policy,
        F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
    {
        using func_result_type = typename detail::then_bulk_function_result<F,
            S, Future, Ts...>::type;

        // std::vector<future<func_result_type>>
        using result_type = std::vector<pika::future<func_result_type>>;

        auto&& func = detail::make_fused_bulk_async_execute_helper<result_type>(
            executor, PIKA_FORWARD(F, f), shape,
            pika::make_tuple(PIKA_FORWARD(Ts, ts)...));

        // void or std::vector<func_result_type>
        using vector_result_type = typename detail::bulk_then_execute_result<F,
            S, Future, Ts...>::type;

        // future<vector_result_type>
        using result_future_type = pika::future<vector_result_type>;

        using shared_state_type =
            typename pika::traits::detail::shared_state_ptr<
                vector_result_type>::type;

        using future_type = typename std::decay<Future>::type;

        // vector<future<func_result_type>> -> vector<func_result_type>
        shared_state_type p = pika::lcos::detail::make_continuation_exec_policy<
            vector_result_type>(PIKA_FORWARD(Future, predecessor), executor,
            policy,
            [func = PIKA_MOVE(func)](
                future_type&& predecessor) mutable -> vector_result_type {
                // use unwrap directly (instead of lazily) to avoid
                // having to pull in dataflow
                return pika::unwrap(func(PIKA_MOVE(predecessor)));
            });

        return pika::traits::future_access<result_future_type>::create(
            PIKA_MOVE(p));
    }
}}}}    // namespace pika::parallel::execution::detail
