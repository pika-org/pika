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
#include <pika/barrier.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/runtime.hpp>
#include <pika/testing/performance.hpp>
#include <pika/thread.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
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

// The "task" method simply spawns total_tasks independent tasks without any special consideration
// for grouping or affinity.
double do_work_task(
    high_resolution_timer& timer, std::uint64_t tasks_per_thread, double task_size_s)
{
    auto const num_threads = pika::get_num_worker_threads();
    auto const total_tasks = num_threads * tasks_per_thread;
    auto sched = ex::thread_pool_scheduler{};
    auto spawn = [=]() {
        return ex::schedule(sched) | ex::then(pika::util::detail::bind_front(task, task_size_s)) |
            ex::ensure_started() | ex::drop_value();
    };

    std::vector<decltype(spawn())> senders;
    senders.reserve(total_tasks);

    for (std::uint64_t i = 0; i < total_tasks; ++i) { senders.push_back(spawn()); }

    double const spawn_time_s = timer.elapsed();

    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    return spawn_time_s;
}

// The "task-hierarchical" method spawns total_tasks independent tasks through num_threads helper tasks,
// hinted to run on each worker thread. The helper tasks spawn tasks on their own worker thread to
// reduce contention.
double do_work_task_hierarchical(
    high_resolution_timer& timer, std::uint64_t tasks_per_thread, double task_size_s)
{
    auto const num_threads = pika::get_num_worker_threads();
    auto const total_tasks = num_threads * tasks_per_thread;
    auto sched = ex::thread_pool_scheduler{};
    auto spawn_helper = [=](std::size_t thread_num) {
        auto sched_with_hint =
            ex::with_hint(sched, pika::execution::thread_schedule_hint(thread_num));
        return ex::schedule(sched_with_hint) | ex::let_value([=] {
            auto spawn = [&]() {
                return ex::schedule(sched_with_hint) |
                    ex::then(pika::util::detail::bind_front(task, task_size_s)) |
                    ex::ensure_started() | ex::drop_value();
            };

            std::vector<decltype(spawn())> senders;
            senders.reserve(total_tasks);

            for (std::uint64_t i = 0; i < tasks_per_thread; ++i) { senders.push_back(spawn()); }

            return ex::when_all_vector(std::move(senders));
        });
    };

    std::vector<decltype(spawn_helper(std::size_t{}))> senders;
    senders.reserve(num_threads);

    for (std::uint64_t i = 0; i < num_threads; ++i) { senders.push_back(spawn_helper(i)); }

    double const spawn_time_s = timer.elapsed();

    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    return spawn_time_s;
}

// The "task-yield" method spawns num_threads tasks. Each of the tasks then yields tasks_per_thread
// times to emulate total_tasks being scheduled. This method is similar to the "barrier" method with
// the difference that the tasks do not run in lockstep.
double do_work_task_yield(
    high_resolution_timer& timer, std::uint64_t tasks_per_thread, double task_size_s)
{
    auto const num_threads = pika::get_num_worker_threads();
    auto sched = ex::thread_pool_scheduler{};
    auto work = [=]() {
        for (std::uint64_t i = 0; i < tasks_per_thread; ++i)
        {
            pika::this_thread::yield();
            task(task_size_s);
        }
    };
    auto spawn = [=]() {
        return ex::schedule(sched) | ex::then(work) | ex::ensure_started() | ex::drop_value();
    };

    std::vector<decltype(spawn())> senders;
    senders.reserve(num_threads);

    for (std::uint64_t i = 0; i < num_threads; ++i) { senders.push_back(spawn()); }

    double const spawn_time_s = timer.elapsed();

    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    return spawn_time_s;
}

// The "barrier" method spawns one task per worker thread and uses a barrier run
// the "tasks" in lockstep on each worker thread.
double do_work_barrier(
    high_resolution_timer& timer, std::uint64_t tasks_per_thread, double task_size_s)
{
    auto const num_threads = pika::get_num_worker_threads();
    auto sched = ex::thread_pool_scheduler{};
    pika::barrier b(num_threads);
    auto work = [=, &b]() {
        for (std::uint64_t i = 0; i < tasks_per_thread; ++i)
        {
            b.arrive_and_wait();
            task(task_size_s);
        }
    };
    auto spawn = [=]() {
        return ex::schedule(sched) | ex::then(work) | ex::ensure_started() | ex::drop_value();
    };

    std::vector<decltype(spawn())> senders;
    senders.reserve(num_threads);

    for (std::uint64_t i = 0; i < num_threads; ++i) { senders.push_back(spawn()); }

    double const spawn_time_s = timer.elapsed();

    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    return spawn_time_s;
}

// The "bulk" method sequences each set of num_worker_threads tasks through bulk.
double do_work_bulk(
    high_resolution_timer& timer, std::uint64_t tasks_per_thread, double task_size_s)
{
    auto const num_threads = pika::get_num_worker_threads();
    auto sched = ex::thread_pool_scheduler{};
    auto work = [=](auto) { task(task_size_s); };

    ex::unique_any_sender<> sender{ex::just()};
    for (std::uint64_t i = 0; i < tasks_per_thread; ++i)
    {
        sender = std::move(sender) | ex::continues_on(sched) | ex::bulk(num_threads, work);
        // To avoid stack overflow when connecting, starting, or destroying the operation state,
        // eagerly start the chain of work periodically using ensure_started. The chosen frequency
        // is mostly arbitrary. It's done as often as reasonably possible to make the probability of
        // a stack overflow very low, but not often enough to introduce significant overhead. 100
        // seems like a good compromise.
        if (i % 100 == 0) { sender = ex::ensure_started(std::move(sender)); }
    }

    double const spawn_time_s = timer.elapsed();

    tt::sync_wait(std::move(sender));

    return spawn_time_s;
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    auto const method = vm["method"].as<std::string>();
    auto const tasks_per_thread = vm["tasks-per-thread"].as<std::uint64_t>();
    auto const task_size_min_s = vm["task-size-min-s"].as<double>();
    auto const task_size_max_s = vm["task-size-max-s"].as<double>();
    auto const task_size_growth_factor = vm["task-size-growth-factor"].as<double>();
    auto const target_efficiency = vm["target-efficiency"].as<double>();
    auto const perftest_json = vm["perftest-json"].as<bool>();

    using do_work_type = double(high_resolution_timer&, std::uint64_t, double);
    do_work_type* do_work = [&]() {
        if (method == "task") { return do_work_task; }
        else if (method == "task-hierarchical") { return do_work_task_hierarchical; }
        else if (method == "task-yield") { return do_work_task_yield; }
        else if (method == "barrier") { return do_work_barrier; }
        else if (method == "bulk") { return do_work_bulk; }
        else
        {
            PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "task_size",
                "--method must be \"task\", \"task-hierarchical\", \"task-yield\", \"barrier\", or "
                "\"bulk\" ({} given)",
                method);
        }
    }();

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
        fmt::print(
            "method,num_threads,tasks_per_thread,task_size_s,single_threaded_reference_time_s,"
            "time_s,spawn_time_s,task_overhead_time_s,parallel_efficiency\n");
    }

    double task_size_s = task_size_min_s;
    double efficiency = 0.0;

    do {
        double const single_threaded_reference_time_s = total_tasks * task_size_s;

        high_resolution_timer timer;
        double const spawn_time_s = do_work(timer, tasks_per_thread, task_size_s);
        double const time_s = timer.elapsed();

        efficiency = single_threaded_reference_time_s / time_s / num_threads;
        if (!perftest_json)
        {
            double const task_overhead_time_s =
                (time_s - single_threaded_reference_time_s / num_threads) / tasks_per_thread;
            fmt::print("{},{},{},{:.9f},{:.9f},{:.9f},{:.9f},{:.9f},{:.4f}\n", method, num_threads,
                tasks_per_thread, task_size_s, single_threaded_reference_time_s, time_s,
                spawn_time_s, task_overhead_time_s, efficiency);
        }

        task_size_s *= task_size_growth_factor;
    } while (efficiency < target_efficiency && task_size_s < task_size_max_s);

    if (perftest_json)
    {
        pika::util::detail::json_perf_times t;
        t.add(fmt::format("task_size - thread_pool_scheduler - {}", method), task_size_s);
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
        ("method", value<std::string>()->default_value("task"), "method used to spawn tasks (\"task\", \"task-hierarchical\", \"task-yield\", \"barrier\", or \"bulk\")")
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
