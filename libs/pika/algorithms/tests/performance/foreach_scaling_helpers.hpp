//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/executors/parallel_executor_aggregated.hpp>
#include <pika/local/algorithm.hpp>
#include <pika/local/chrono.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>

#include "worker_timed.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int delay = 1000;
int test_count = 100;
int chunk_size = 0;
int num_overlapping_loops = 0;
bool disable_stealing = false;
bool fast_idle_mode = false;
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

struct disable_stealing_parameter
{
    template <typename Executor>
    void mark_begin_execution(Executor&&)
    {
        pika::threads::add_remove_scheduler_mode(
            pika::threads::policies::enable_stealing,
            pika::threads::policies::enable_idle_backoff);
    }

    template <typename Executor>
    void mark_end_of_scheduling(Executor&&)
    {
        pika::threads::remove_scheduler_mode(
            pika::threads::policies::enable_stealing);
    }

    template <typename Executor>
    void mark_end_execution(Executor&&)
    {
        pika::threads::add_remove_scheduler_mode(
            pika::threads::policies::enable_idle_backoff,
            pika::threads::policies::enable_stealing);
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<disable_stealing_parameter> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void measure_plain_for(std::vector<std::size_t> const& data_representation)
{
    std::size_t num = data_representation.size();

    std::size_t size = num & std::size_t(-4);    // -V112
    for (std::size_t i = 0; i < size; i += 4)
    {
        worker_timed(delay);
        worker_timed(delay);
        worker_timed(delay);
        worker_timed(delay);
    }
    for (/**/; size < num; ++size)
    {
        worker_timed(delay);
    }
}

void measure_plain_for_iter(std::vector<std::size_t> const& data_representation)
{
    for (auto&& v : data_representation)
    {
        PIKA_UNUSED(v);
        worker_timed(delay);
    }
}

///////////////////////////////////////////////////////////////////////////////
void measure_sequential_foreach(
    std::vector<std::size_t> const& data_representation)
{
    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke sequential for_each
        pika::ranges::for_each(pika::execution::seq.with(dsp),
            data_representation, [](std::size_t) { worker_timed(delay); });
    }
    else
    {
        // invoke sequential for_each
        pika::ranges::for_each(pika::execution::seq, data_representation,
            [](std::size_t) { worker_timed(delay); });
    }
}

template <typename Executor>
void measure_parallel_foreach(
    std::vector<std::size_t> const& data_representation, Executor&& exec)
{
    // create executor parameters object
    pika::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_each
        pika::ranges::for_each(pika::execution::par.with(cs, dsp).on(exec),
            data_representation, [](std::size_t) { worker_timed(delay); });
    }
    else
    {
        // invoke parallel for_each
        pika::ranges::for_each(pika::execution::par.with(cs).on(exec),
            data_representation, [](std::size_t) { worker_timed(delay); });
    }
}

template <typename Executor>
pika::future<void> measure_task_foreach(
    std::shared_ptr<std::vector<std::size_t>> data_representation,
    Executor&& exec)
{
    // create executor parameters object
    pika::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_each
        return pika::ranges::for_each(
            pika::execution::par(pika::execution::task).with(cs, dsp).on(exec),
            *data_representation, [](std::size_t) { worker_timed(delay); })
            .then([data_representation](pika::future<void>) {});
    }
    else
    {
        // invoke parallel for_each
        return pika::ranges::for_each(
            pika::execution::par(pika::execution::task).with(cs).on(exec),
            *data_representation, [](std::size_t) { worker_timed(delay); })
            .then([data_representation](pika::future<void>) {});
    }
}

///////////////////////////////////////////////////////////////////////////////
void measure_sequential_forloop(
    std::vector<std::size_t> const& data_representation)
{
    using iterator = typename std::vector<std::size_t>::const_iterator;

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke sequential for_loop
        pika::for_loop(pika::execution::seq.with(dsp),
            std::begin(data_representation), std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
    else
    {
        // invoke sequential for_loop
        pika::for_loop(pika::execution::seq, std::begin(data_representation),
            std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
}

template <typename Executor>
void measure_parallel_forloop(
    std::vector<std::size_t> const& data_representation, Executor&& exec)
{
    using iterator = typename std::vector<std::size_t>::const_iterator;

    // create executor parameters object
    pika::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_loop
        pika::for_loop(pika::execution::par.with(cs, dsp).on(exec),
            std::begin(data_representation), std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
    else
    {
        // invoke parallel for_loop
        pika::for_loop(pika::execution::par.with(cs).on(exec),
            std::begin(data_representation), std::end(data_representation),
            [](iterator) { worker_timed(delay); });
    }
}

template <typename Executor>
pika::future<void> measure_task_forloop(
    std::shared_ptr<std::vector<std::size_t>> data_representation,
    Executor&& exec)
{
    using iterator = typename std::vector<std::size_t>::const_iterator;

    // create executor parameters object
    pika::execution::static_chunk_size cs(chunk_size);

    if (disable_stealing)
    {
        // disable stealing from inside the algorithm
        disable_stealing_parameter dsp;

        // invoke parallel for_loop
        return pika::for_loop(
            pika::execution::par(pika::execution::task).with(cs, dsp).on(exec),
            std::begin(*data_representation), std::end(*data_representation),
            [](iterator) { worker_timed(delay); })
            .then([data_representation](pika::future<void>) {});
    }
    else
    {
        // invoke parallel for_loop
        return pika::for_loop(
            pika::execution::par(pika::execution::task).with(cs).on(exec),
            std::begin(*data_representation), std::end(*data_representation),
            [](iterator) { worker_timed(delay); })
            .then([data_representation](pika::future<void>) {});
    }
}
#endif
