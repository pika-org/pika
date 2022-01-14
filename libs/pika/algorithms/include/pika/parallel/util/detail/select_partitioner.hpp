//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/futures/future.hpp>

#include <pika/executors/execution_policy.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace util { namespace detail {
    template <typename ExPolicy, template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner
    {
        template <typename... Args>
        using apply = Partitioner<ExPolicy, Args...>;
    };

    template <template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner<pika::execution::parallel_task_policy, Partitioner,
        TaskPartitioner>
    {
        template <typename... Args>
        using apply =
            TaskPartitioner<pika::execution::parallel_task_policy, Args...>;
    };

    template <typename Executor, typename Parameters,
        template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner<
        pika::execution::parallel_task_policy_shim<Executor, Parameters>,
        Partitioner, TaskPartitioner>
    {
        template <typename... Args>
        using apply = TaskPartitioner<
            pika::execution::parallel_task_policy_shim<Executor, Parameters>,
            Args...>;
    };

#if defined(PIKA_HAVE_DATAPAR)
    template <template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner<pika::execution::par_simd_task_policy, Partitioner,
        TaskPartitioner>
    {
        template <typename... Args>
        using apply =
            TaskPartitioner<pika::execution::par_simd_task_policy, Args...>;
    };
#endif
}}}}    // namespace pika::parallel::util::detail
