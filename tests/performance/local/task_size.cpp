//  Copyright (c) 2018-2023 ETH Zurich
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test finds the smallest task size that yields a parallel efficiency at least as large as the
// target efficiency. The timing is always compared to an ideal single-threaded execution, i.e. the
// number of tasks multiplied by the task size. The search starts from the minimum task size. The
// task size is then grown geometrically until the efficiency of the parallel execution of the given
// number of tasks is above the target efficiency.
//
// This test is useful for finding an appropriate minimum task size for a system, i.e. one that
// gives reasonable parallel efficiency.

#include <pika/config.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/runtime.hpp>
#include <pika/testing/performance.hpp>
#include <pika/thread.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

using pika::program_options::bool_switch;
using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::chrono::detail::high_resolution_timer;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

void task(double task_size_s) noexcept
{
    high_resolution_timer t;
    pika::util::yield_while([&]() { return t.elapsed() < task_size_s; }, nullptr, false);
}

void do_work(std::uint64_t tasks, double task_size_s)
{
    auto sched = ex::thread_pool_scheduler{};
    auto spawn = [=]() {
        return ex::schedule(sched) | ex::then(pika::util::detail::bind_front(task, task_size_s)) |
            ex::ensure_started() | ex::drop_value();
    };

    std::vector<decltype(spawn())> senders;
    senders.reserve(tasks);

    for (std::uint64_t i = 0; i < tasks; ++i) { senders.push_back(spawn()); }
    tt::sync_wait(ex::when_all_vector(std::move(senders)));
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    auto const tasks_per_thread = vm["tasks-per-thread"].as<std::uint64_t>();
    auto const task_size_min_s = vm["task-size-min-s"].as<double>();
    auto const task_size_max_s = vm["task-size-max-s"].as<double>();
    auto const task_size_growth_factor = vm["task-size-growth-factor"].as<double>();
    auto const target_efficiency = vm["target-efficiency"].as<double>();
    auto const perftest_json = vm["perftest-json"].as<bool>();

    if (task_size_min_s <= 0)
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "task_size",
            "--task-size-min-s must be strictly larger than zero ({} given)", task_size_min_s);
    }

    if (task_size_max_s <= 0)
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "task_size",
            "--task-size-max-s must be strictly larger than zero ({} given)", task_size_max_s);
    }

    if (task_size_max_s <= task_size_min_s)
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "task_size",
            "--task-size-max-s must be strictly larger than --task-size-min-s ({} and {} given, "
            "respectively)",
            task_size_max_s, task_size_min_s);
    }

    if (task_size_growth_factor <= 1)
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "task_size",
            "--task-size-growth-factor must be strictly larger than one ({} given)",
            task_size_growth_factor);
    }

    if (target_efficiency <= 0 || target_efficiency >= 1)
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "task_size",
            "--target-efficiency must be strictly between 0 and 1 ({} given)", target_efficiency);
    }

    auto const num_threads = pika::get_num_worker_threads();
    auto const total_tasks = num_threads * tasks_per_thread;

    if (!perftest_json)
    {
        fmt::print("num_threads,tasks_per_thread,task_size_s,single_threaded_reference_time,time,"
                   "parallel_efficiency\n");
    }

    double task_size_s = task_size_min_s;
    double efficiency = 0.0;

    do {
        double const single_threaded_reference_time = total_tasks * task_size_s;

        high_resolution_timer timer;
        do_work(total_tasks, task_size_s);
        double time = timer.elapsed();

        efficiency = single_threaded_reference_time / time / num_threads;
        if (!perftest_json)
        {
            fmt::print("{},{},{:.9f},{:.9f},{:.9f},{:.4f}\n", num_threads, tasks_per_thread,
                task_size_s, single_threaded_reference_time, time, efficiency);
        }

        task_size_s *= task_size_growth_factor;
    } while (efficiency < target_efficiency && task_size_s < task_size_max_s);

    if (perftest_json)
    {
        pika::util::detail::json_perf_times t;
        t.add("task_size - thread_pool_scheduler", task_size_s);
        std::cout << t;
    }

    pika::finalize();
    return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");
    // clang-format off
    cmdline.add_options()
        ("tasks-per-thread", value<std::uint64_t>()->default_value(1000), "number of tasks to invoke per thread")
        ("task-size-min-s", value<double>()->default_value(1e-6), "initial task size in seconds")
        ("task-size-max-s", value<double>()->default_value(1e-2), "maximum task size in seconds at which to stop the test")
        ("task-size-growth-factor", value<double>()->default_value(1.5), "factor with which to grow the task size each iteration")
        ("target-efficiency", value<double>()->default_value(0.90), "target parallel efficiency at which to stop the test")
        ("perftest-json", bool_switch(), "print final task size in json format for use with performance CI.")
        // clang-format on
        ;

    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;
    return pika::init(pika_main, argc, argv, init_args);
}
