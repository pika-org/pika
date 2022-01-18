//  Copyright (c) 2018-2020 Mikael Simberg
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/runtime.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/synchronization.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/timing.hpp>
#include <pika/threading_base/annotated_function.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::apply;
using pika::async;
using pika::future;

using pika::chrono::high_resolution_timer;

// global vars we stick here to make printouts easy for plotting
static std::string queuing = "default";
static std::size_t numa_sensitive = 0;
static std::uint64_t num_threads = 1;
static std::string info_string = "";

///////////////////////////////////////////////////////////////////////////////
void print_stats(const char* title, const char* wait, const char* exec,
    std::int64_t count, double duration, bool csv)
{
    std::ostringstream temp;
    double us = 1e6 * duration / count;
    if (csv)
    {
        pika::util::format_to(temp,
            "{1}, {:27}, {:15}, {:18}, {:8}, {:8}, {:20}, {:4}, {:4}, "
            "{:20}",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string);
    }
    else
    {
        pika::util::format_to(temp,
            "invoked {:1}, futures {:27} {:15} {:18} in {:8} seconds : {:8} "
            "us/future, queue {:20}, numa {:4}, threads {:4}, info {:20}",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string);
    }
    std::cout << temp.str() << std::endl;
    // CDash graph plotting
    //pika::util::print_cdash_timing(title, duration);
}

const char* exec_name(pika::execution::parallel_executor const&)
{
    return "parallel_executor";
}

const char* exec_name(
    pika::parallel::execution::parallel_executor_aggregated const&)
{
    return "parallel_executor_aggregated";
}

const char* exec_name(pika::execution::experimental::scheduler_executor<
    pika::execution::experimental::thread_pool_scheduler> const&)
{
    return "scheduler_executor<thread_pool_scheduler>";
}

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double null_function() noexcept
{
    if (num_iterations > 0)
    {
        const int array_size = 4096;
        std::array<double, array_size> dummy;
        for (std::uint64_t i = 0; i < num_iterations; ++i)
        {
            for (std::uint64_t j = 0; j < array_size; ++j)
            {
                dummy[j] = 1.0 / (2.0 * i * j + 1.0);
            }
        }
        return dummy[0];
    }
    return 0.0;
}

struct scratcher
{
    void operator()(future<double> r) const
    {
        global_scratch += r.get();
    }
};

// Time async execution using wait each on futures vector
template <typename Executor>
void measure_function_futures_wait_each(
    std::uint64_t count, bool csv, Executor& exec)
{
    std::vector<future<double>> futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(exec, &null_function));
    pika::wait_each(scratcher(), futures);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("async", "WaitEach", exec_name(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_wait_all(
    std::uint64_t count, bool csv, Executor& exec)
{
    std::vector<future<double>> futures;
    futures.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
        futures.push_back(async(exec, &null_function));
    pika::wait_all(futures);

    const double duration = walltime.elapsed();
    print_stats("async", "WaitAll", exec_name(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_limiting_executor(
    std::uint64_t count, bool csv, Executor exec)
{
    std::uint64_t const num_threads = pika::get_num_worker_threads();
    std::uint64_t const tasks = num_threads * 2000;
    std::atomic<std::uint64_t> sanity_check(count);

    auto const sched = pika::threads::get_self_id_data()->get_scheduler_base();
    if (std::string("core-shared_priority_queue_scheduler") ==
        sched->get_description())
    {
        sched->add_remove_scheduler_mode(
            // add these flags
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::enable_stealing |
                pika::threads::policies::assign_work_round_robin |
                pika::threads::policies::steal_after_local),
            // remove these flags
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::enable_stealing_numa |
                pika::threads::policies::assign_work_thread_parent |
                pika::threads::policies::steal_high_priority_first));
    }

    // test a parallel algorithm on custom pool with high priority
    auto const chunk_size = count / (num_threads * 2);
    pika::execution::static_chunk_size fixed(chunk_size);

    // start the clock
    high_resolution_timer walltime;
    {
        pika::execution::experimental::limiting_executor<Executor> signal_exec(
            exec, tasks, tasks + 1000);
        pika::for_loop(
            pika::execution::par.with(fixed), 0, count, [&](std::uint64_t) {
                pika::apply(signal_exec, [&]() {
                    null_function();
                    sanity_check--;
                });
            });
    }

    if (sanity_check != 0)
    {
        throw std::runtime_error(
            "This test is faulty " + std::to_string(sanity_check));
    }

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats(
        "apply", "limiting-Exec", exec_name(exec), count, duration, csv);
}

template <typename Executor>
void measure_function_futures_sliding_semaphore(
    std::uint64_t count, bool csv, Executor& exec)
{
    // start the clock
    high_resolution_timer walltime;
    const int sem_count = 5000;
    pika::lcos::local::sliding_semaphore sem(sem_count);
    for (std::uint64_t i = 0; i < count; ++i)
    {
        pika::async(exec, [i, &sem]() {
            null_function();
            sem.signal(i);
        });
        sem.wait(i);
    }
    sem.wait(count + sem_count - 1);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("apply", "Sliding-Sem", exec_name(exec), count, duration, csv);
}

struct unlimited_number_of_chunks
{
    template <typename Executor>
    std::size_t maximal_number_of_chunks(
        Executor&& /*executor*/, std::size_t /*cores*/, std::size_t num_tasks)
    {
        return num_tasks;
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<unlimited_number_of_chunks> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

template <typename Executor>
void measure_function_futures_for_loop(std::uint64_t count, bool csv,
    Executor& exec, char const* executor_name = nullptr)
{
    // start the clock
    high_resolution_timer walltime;
    pika::for_loop(
        pika::execution::par.on(exec).with(
            pika::execution::static_chunk_size(1), unlimited_number_of_chunks()),
        0, count, [](std::uint64_t) { null_function(); });

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("for_loop", "par",
        executor_name ? executor_name : exec_name(exec), count, duration, csv);
}

void measure_function_futures_register_work(std::uint64_t count, bool csv)
{
    pika::lcos::local::latch l(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        pika::threads::thread_init_data data(
            pika::threads::make_thread_function_nullary([&l]() {
                null_function();
                l.count_down(1);
            }),
            "null_function");
        pika::threads::register_work(data);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("register_work", "latch", "none", count, duration, csv);
}

void measure_function_futures_create_thread(std::uint64_t count, bool csv)
{
    pika::lcos::local::latch l(count);

    auto const sched = pika::threads::get_self_id_data()->get_scheduler_base();
    auto func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const thread_func =
        pika::threads::detail::thread_function_nullary<decltype(func)>{func};
    auto const desc = pika::util::thread_description();
    auto const prio = pika::threads::thread_priority::normal;
    auto const hint = pika::threads::thread_schedule_hint();
    auto const stack_size = pika::threads::thread_stacksize::small_;
    pika::error_code ec;

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        auto init = pika::threads::thread_init_data(
            pika::threads::thread_function_type(thread_func), desc, prio, hint,
            stack_size, pika::threads::thread_schedule_state::pending, false,
            sched);
        sched->create_thread(init, nullptr, ec);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("create_thread", "latch", "none", count, duration, csv);
}

void measure_function_futures_create_thread_hierarchical_placement(
    std::uint64_t count, bool csv)
{
    pika::lcos::local::latch l(count);

    auto sched = pika::threads::get_self_id_data()->get_scheduler_base();

    if (std::string("core-shared_priority_queue_scheduler") ==
        sched->get_description())
    {
        sched->add_remove_scheduler_mode(
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::assign_work_thread_parent),
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::enable_stealing |
                pika::threads::policies::enable_stealing_numa |
                pika::threads::policies::assign_work_round_robin |
                pika::threads::policies::steal_after_local |
                pika::threads::policies::steal_high_priority_first));
    }
    auto const func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const thread_func =
        pika::threads::detail::thread_function_nullary<decltype(func)>{func};
    auto const desc = pika::util::thread_description();
    auto prio = pika::threads::thread_priority::normal;
    auto const stack_size = pika::threads::thread_stacksize::small_;
    auto const num_threads = pika::get_num_worker_threads();
    pika::error_code ec;

    // start the clock
    high_resolution_timer walltime;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto const hint =
            pika::threads::thread_schedule_hint(static_cast<std::int16_t>(t));
        auto spawn_func = [&thread_func, sched, hint, t, count, num_threads,
                              desc, prio]() {
            std::uint64_t const count_start = t * count / num_threads;
            std::uint64_t const count_end = (t + 1) * count / num_threads;
            pika::error_code ec;
            for (std::uint64_t i = count_start; i < count_end; ++i)
            {
                pika::threads::thread_init_data init(
                    pika::threads::thread_function_type(thread_func), desc, prio,
                    hint, stack_size,
                    pika::threads::thread_schedule_state::pending, false, sched);
                sched->create_thread(init, nullptr, ec);
            }
        };
        auto const thread_spawn_func =
            pika::threads::detail::thread_function_nullary<decltype(spawn_func)>{
                spawn_func};

        pika::threads::thread_init_data init(
            pika::threads::thread_function_type(thread_spawn_func), desc, prio,
            hint, stack_size, pika::threads::thread_schedule_state::pending,
            false, sched);
        sched->create_thread(init, nullptr, ec);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats(
        "create_thread_hierarchical", "latch", "none", count, duration, csv);
}

void measure_function_futures_apply_hierarchical_placement(
    std::uint64_t count, bool csv)
{
    pika::lcos::local::latch l(count);

    auto const func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const num_threads = pika::get_num_worker_threads();

    // start the clock
    high_resolution_timer walltime;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto const hint =
            pika::threads::thread_schedule_hint(static_cast<std::int16_t>(t));
        auto spawn_func = [&func, hint, t, count, num_threads]() {
            auto exec = pika::execution::parallel_executor(hint);
            std::uint64_t const count_start = t * count / num_threads;
            std::uint64_t const count_end = (t + 1) * count / num_threads;

            for (std::uint64_t i = count_start; i < count_end; ++i)
            {
                pika::apply(exec, func);
            }
        };

        auto exec = pika::execution::parallel_executor(hint);
        pika::apply(exec, spawn_func);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("apply_hierarchical", "latch", "parallel_executor", count,
        duration, csv);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    {
        if (vm.count("pika:queuing"))
            queuing = vm["pika:queuing"].as<std::string>();

        if (vm.count("pika:numa-sensitive"))
            numa_sensitive = 1;
        else
            numa_sensitive = 0;

        bool test_all = (vm.count("test-all") > 0);
        const int repetitions = vm["repetitions"].as<int>();

        if (vm.count("info"))
            info_string = vm["info"].as<std::string>();

        num_threads = pika::get_num_worker_threads();

        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const std::uint64_t count = vm["futures"].as<std::uint64_t>();
        bool csv = vm.count("csv") != 0;
        if (PIKA_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        pika::execution::parallel_executor par;
        pika::parallel::execution::parallel_executor_aggregated par_agg;
        pika::execution::parallel_executor par_nostack(
            pika::threads::thread_priority::default_,
            pika::threads::thread_stacksize::nostack);
        pika::execution::experimental::scheduler_executor<
            pika::execution::experimental::thread_pool_scheduler>
            sched_exec_tps;

        for (int i = 0; i < repetitions; i++)
        {
            measure_function_futures_create_thread_hierarchical_placement(
                count, csv);
            if (test_all)
            {
                measure_function_futures_limiting_executor(count, csv, par);
                measure_function_futures_wait_each(count, csv, par);
                measure_function_futures_wait_all(count, csv, par);
                measure_function_futures_sliding_semaphore(count, csv, par);
                measure_function_futures_for_loop(count, csv, par);
                measure_function_futures_for_loop(count, csv, par_agg);
                measure_function_futures_for_loop(count, csv, sched_exec_tps);
                measure_function_futures_for_loop(
                    count, csv, par_nostack, "parallel_executor_nostack");
                measure_function_futures_register_work(count, csv);
                measure_function_futures_create_thread(count, csv);
                measure_function_futures_apply_hierarchical_placement(
                    count, csv);
            }
        }
    }

    return pika::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()("futures",
        value<std::uint64_t>()->default_value(500000),
        "number of futures to invoke")

        ("delay-iterations", value<std::uint64_t>()->default_value(0),
         "number of iterations in the delay loop")

        ("csv", "output results as csv (format: count,duration)")
        ("test-all", "run all benchmarks")
        ("repetitions", value<int>()->default_value(1),
         "number of repetitions of the full benchmark")

        ("info", value<std::string>()->default_value("no-info"),
         "extra info for plot output (e.g. branch name)");
    // clang-format on

    // Initialize and run pika.
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::local::init(pika_main, argc, argv, init_args);
}
