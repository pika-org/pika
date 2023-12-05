//  Copyright (c) 2018-2023 ETH Zurich
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/latch.hpp>
#include <pika/modules/synchronization.hpp>
#include <pika/modules/timing.hpp>
#include <pika/runtime.hpp>
#include <pika/thread.hpp>
#include <pika/threading_base/annotated_function.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::chrono::detail::high_resolution_timer;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

// global vars we stick here to make printouts easy for plotting
static std::string queuing = "default";
static std::size_t numa_sensitive = 0;
static std::uint64_t num_threads = 1;
static std::string info_string = "";

///////////////////////////////////////////////////////////////////////////////
void print_stats(const char* title, const char* wait, const char* sched, std::int64_t count,
    double duration, bool csv)
{
    std::ostringstream temp;
    double us = 1e6 * duration / count;
    if (csv)
    {
        fmt::print(temp, "{}, {:27}, {:15}, {:45}, {:8}, {:8}, {:20}, {:4}, {:4}, {:20}", count,
            title, wait, sched, duration, us, queuing, numa_sensitive, num_threads, info_string);
    }
    else
    {
        fmt::print(temp,
            "invoked {:1}, tasks {:27} {:15} {:18} in {:8} seconds : {:8} us/task, queue "
            "{:20}, numa {:4}, threads {:4}, info {:20}",
            count, title, wait, sched, duration, us, queuing, numa_sensitive, num_threads,
            info_string);
    }
    std::cout << temp.str() << std::endl;
    // CDash graph plotting
    //pika::util::print_cdash_timing(title, duration);
}

const char* sched_name(ex::thread_pool_scheduler const&) { return "thread_pool_scheduler"; }

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
            for (std::uint64_t j = 0; j < array_size; ++j) { dummy[j] = 1.0 / (2.0 * i * j + 1.0); }
        }
        return dummy[0];
    }
    return 0.0;
}

struct scratcher
{
    void operator()(double r) const { global_scratch += r; }
};

template <typename Scheduler>
void function_senders_when_all_vector(std::uint64_t count, bool csv, Scheduler& sched)
{
    auto spawn = [&]() { return ex::schedule(sched) | ex::then(null_function) | ex::drop_value(); };
    std::vector<decltype(spawn())> senders;
    senders.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i) senders.push_back(spawn());
    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("schedule", "WhenAll", sched_name(sched), count, duration, csv);
}

template <typename Scheduler>
void function_senders_when_all_vector_eager(std::uint64_t count, bool csv, Scheduler& sched)
{
    auto spawn = [&]() {
        return ex::schedule(sched) | ex::then(null_function) | ex::ensure_started() |
            ex::drop_value();
    };
    std::vector<decltype(spawn())> senders;
    senders.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i) senders.push_back(spawn());
    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("schedule", "WhenAllEager", sched_name(sched), count, duration, csv);
}

template <typename Scheduler>
void function_senders_when_all_vector_any_sender(std::uint64_t count, bool csv, Scheduler& sched)
{
    auto spawn = [&]() { return ex::schedule(sched) | ex::then(null_function) | ex::drop_value(); };
    std::vector<ex::unique_any_sender<>> senders;
    senders.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i) senders.emplace_back(spawn());
    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("schedule", "WhenAllAnySender", sched_name(sched), count, duration, csv);
}

template <typename Scheduler>
void function_senders_when_all_vector_eager_any_sender(
    std::uint64_t count, bool csv, Scheduler& sched)
{
    auto spawn = [&]() {
        return ex::schedule(sched) | ex::then(null_function) | ex::ensure_started() |
            ex::drop_value();
    };
    std::vector<ex::unique_any_sender<>> senders;
    senders.reserve(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i) senders.emplace_back(spawn());
    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("schedule", "WhenAllEagerAnySender", sched_name(sched), count, duration, csv);
}

template <typename Scheduler>
void function_sliding_semaphore_execute(std::uint64_t count, bool csv, Scheduler& sched)
{
    // start the clock
    high_resolution_timer walltime;
    const int sem_count = 5000;
    pika::sliding_semaphore sem(sem_count);
    for (std::uint64_t i = 0; i < count; ++i)
    {
        ex::execute(sched, [i, &sem]() {
            null_function();
            sem.signal(i);
        });
        sem.wait(i);
    }
    sem.wait(count + sem_count - 1);

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("execute", "Sliding-Sem", sched_name(sched), count, duration, csv);
}

void function_register_work(std::uint64_t count, bool csv)
{
    pika::latch l(count);

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        pika::threads::detail::thread_init_data data(
            pika::threads::detail::make_thread_function_nullary([&l]() {
                null_function();
                l.count_down(1);
            }),
            "null_function");
        pika::threads::detail::register_work(data);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("register_work", "latch", "none", count, duration, csv);
}

void function_create_thread(std::uint64_t count, bool csv)
{
    pika::latch l(count);

    auto const sched = pika::threads::detail::get_self_id_data()->get_scheduler_base();
    auto func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const thread_func = pika::threads::detail::thread_function_nullary<decltype(func)>{func};
    auto const desc = pika::detail::thread_description();
    auto const prio = pika::execution::thread_priority::normal;
    auto const hint = pika::execution::thread_schedule_hint();
    auto const stack_size = pika::execution::thread_stacksize::small_;
    pika::error_code ec;

    // start the clock
    high_resolution_timer walltime;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        auto init = pika::threads::detail::thread_init_data(
            pika::threads::detail::thread_function_type(thread_func), desc, prio, hint, stack_size,
            pika::threads::detail::thread_schedule_state::pending, false, sched);
        sched->create_thread(init, nullptr, ec);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("create_thread", "latch", "none", count, duration, csv);
}

void function_create_thread_hierarchical_placement(std::uint64_t count, bool csv)
{
    pika::latch l(count);

    auto sched = pika::threads::detail::get_self_id_data()->get_scheduler_base();

    if (std::string("core-shared_priority_queue_scheduler") == sched->get_description())
    {
        using ::pika::threads::scheduler_mode;
        sched->add_scheduler_mode(scheduler_mode::assign_work_thread_parent);
        sched->remove_scheduler_mode(scheduler_mode::enable_stealing |
            scheduler_mode::enable_stealing_numa | scheduler_mode::assign_work_round_robin |
            scheduler_mode::steal_after_local | scheduler_mode::steal_high_priority_first);
    }
    auto const func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const thread_func = pika::threads::detail::thread_function_nullary<decltype(func)>{func};
    auto const desc = pika::detail::thread_description();
    auto prio = pika::execution::thread_priority::normal;
    auto const stack_size = pika::execution::thread_stacksize::small_;
    auto const num_threads = pika::get_num_worker_threads();
    pika::error_code ec;

    // start the clock
    high_resolution_timer walltime;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto const hint = pika::execution::thread_schedule_hint(static_cast<std::int16_t>(t));
        auto spawn_func = [&thread_func, sched, hint, t, count, num_threads, desc, prio]() {
            std::uint64_t const count_start = t * count / num_threads;
            std::uint64_t const count_end = (t + 1) * count / num_threads;
            pika::error_code ec;
            for (std::uint64_t i = count_start; i < count_end; ++i)
            {
                pika::threads::detail::thread_init_data init(
                    pika::threads::detail::thread_function_type(thread_func), desc, prio, hint,
                    stack_size, pika::threads::detail::thread_schedule_state::pending, false,
                    sched);
                sched->create_thread(init, nullptr, ec);
            }
        };
        auto const thread_spawn_func =
            pika::threads::detail::thread_function_nullary<decltype(spawn_func)>{spawn_func};

        pika::threads::detail::thread_init_data init(
            pika::threads::detail::thread_function_type(thread_spawn_func), desc, prio, hint,
            stack_size, pika::threads::detail::thread_schedule_state::pending, false, sched);
        sched->create_thread(init, nullptr, ec);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("create_thread_hierarchical", "latch", "none", count, duration, csv);
}

void function_apply_hierarchical_placement(std::uint64_t count, bool csv)
{
    pika::latch l(count);

    auto const func = [&l]() {
        null_function();
        l.count_down(1);
    };
    auto const num_threads = pika::get_num_worker_threads();

    // start the clock
    high_resolution_timer walltime;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto const hint = pika::execution::thread_schedule_hint(static_cast<std::int16_t>(t));
        auto spawn_func = [&func, hint, t, count, num_threads]() {
            auto sched = ex::with_hint(ex::thread_pool_scheduler{}, hint);
            std::uint64_t const count_start = t * count / num_threads;
            std::uint64_t const count_end = (t + 1) * count / num_threads;

            for (std::uint64_t i = count_start; i < count_end; ++i) { ex::execute(sched, func); }
        };

        auto sched = ex::with_hint(ex::thread_pool_scheduler{}, hint);
        ex::execute(sched, spawn_func);
    }
    l.wait();

    // stop the clock
    const double duration = walltime.elapsed();
    print_stats("execute_hierarchical", "latch", "thread_pool_scheduler", count, duration, csv);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    {
        if (vm.count("pika:queuing")) queuing = vm["pika:queuing"].as<std::string>();

        if (vm.count("pika:numa-sensitive"))
            numa_sensitive = 1;
        else
            numa_sensitive = 0;

        bool test_all = (vm.count("test-all") > 0);
        const int repetitions = vm["repetitions"].as<int>();

        if (vm.count("info")) info_string = vm["info"].as<std::string>();

        num_threads = pika::get_num_worker_threads();

        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const std::uint64_t count = vm["tasks"].as<std::uint64_t>();
        bool csv = vm.count("csv") != 0;
        if (PIKA_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 tasks specified\n");

        ex::thread_pool_scheduler sched;

        for (int i = 0; i < repetitions; i++)
        {
            function_create_thread_hierarchical_placement(count, csv);
            if (test_all)
            {
                function_senders_when_all_vector(count, csv, sched);
                function_senders_when_all_vector_eager(count, csv, sched);
                function_senders_when_all_vector_any_sender(count, csv, sched);
                function_senders_when_all_vector_eager_any_sender(count, csv, sched);
                // function_sliding_semaphore_execute(count, csv, sched);
                function_register_work(count, csv);
                function_create_thread(count, csv);
                function_apply_hierarchical_placement(count, csv);
            }
        }
    }

    pika::finalize();
    return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()("tasks",
        value<std::uint64_t>()->default_value(500000),
        "number of tasks to invoke")

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
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
