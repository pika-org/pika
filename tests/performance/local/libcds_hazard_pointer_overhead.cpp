//  Copyright (c) 2020 Weile Wei
//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <pika/pika_init.hpp>
#include <pika/include/apply.hpp>
#include <pika/include/async.hpp>
#include <pika/include/parallel_executors.hpp>
#include <pika/include/parallel_for_loop.hpp>
#include <pika/include/threads.hpp>
#include <pika/iostream.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/timing.hpp>
#include <pika/threading_base/annotated_function.hpp>

#include <pika/include/parallel_execution.hpp>
#include <pika/modules/synchronization.hpp>

#include <cds/gc/hp.h>    // for cds::HP (Hazard Pointer) SMR
#include <cds/init.h>     // for cds::Initialize and cds::Terminate

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

using pika::find_here;
using pika::naming::id_type;

using pika::apply;
using pika::async;
using pika::future;

using pika::chrono::high_resolution_timer;

using pika::cout;
using pika::flush;

// global vars we stick here to make printouts easy for plotting
static std::string queuing = "default";
static std::size_t numa_sensitive = 0;
static std::uint64_t num_threads = 1;
static std::string info_string = "";

///////////////////////////////////////////////////////////////////////////////
void print_stats(const char* title, const char* wait, const char* exec,
    std::int64_t count, double duration, bool csv, bool libcds)
{
    std::ostringstream temp;
    double us = 1e6 * duration / count;
    if (csv)
    {
        pika::util::format_to(temp,
            "{1}, {:27}, {:15}, {:28}, {:8}, {:8}, {:20}, {:4}, {:4}, "
            "{:20}, {:4}",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string, libcds);
    }
    else
    {
        pika::util::format_to(temp,
            "invoked {:1} futures, {:27} {:15} {:28} in {:8} seconds : {:8} "
            "us/future, queue {:20}, numa {:4}, threads {:4}, info {:20}"
            ", libcds {:4}",
            count, title, wait, exec, duration, us, queuing, numa_sensitive,
            num_threads, info_string, libcds);
    }
    std::cout << temp.str() << std::endl;
    // CDash graph plotting
    //pika::util::print_cdash_timing(title, duration);
}

const char* exec_name(pika::execution::parallel_executor const& exec)
{
    return "parallel_executor";
}

const char* exec_name(
    pika::parallel::execution::parallel_executor_aggregated const& exec)
{
    return "parallel_executor_aggregated";
}

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
struct libcds_thread_manager_wrapper
{
    explicit libcds_thread_manager_wrapper(bool uselibcds)
      : uselibcds_(uselibcds)
    {
        if (uselibcds_)
            cds::gc::hp::smr::attach_thread();
    }
    ~libcds_thread_manager_wrapper()
    {
        if (uselibcds_)
            cds::gc::hp::smr::detach_thread();
    }

    bool uselibcds_;
};

double null_function(bool uselibcds) noexcept
{
    libcds_thread_manager_wrapper wrap(uselibcds);

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

void measure_function_futures_create_thread_hierarchical_placement(
    std::uint64_t count, bool csv, bool uselibcds)
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
    auto const func = [&]() {
        null_function(uselibcds);
        l.count_down(1);
    };
    auto const thread_func =
        pika::threads::detail::thread_function_nullary<decltype(func)>{func};
    auto desc = pika::util::thread_description();
    auto prio = pika::threads::thread_priority::normal;
    auto stack_size = pika::threads::thread_stacksize::small_;
    auto num_threads = pika::get_num_worker_threads();
    pika::error_code ec;

    // start the clock
    high_resolution_timer walltime;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto const hint =
            pika::threads::thread_schedule_hint(static_cast<std::int16_t>(t));
        auto spawn_func = [&thread_func, sched, hint, t, count, num_threads,
                              desc, prio, stack_size]() {
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
    print_stats("create_thread_hierarchical", "latch", "none", count, duration,
        csv, uselibcds);
}

///////////////////////////////////////////////////////////////////////////////
struct libcds_wrapper
{
    libcds_wrapper()
    {
        // Initialize libcds
        cds::Initialize();
    }

    ~libcds_wrapper()
    {
        // Terminate libcds
        cds::Terminate();
    }
};

int pika_main(variables_map& vm)
{
    // Initialize libcds
    libcds_wrapper wrapper;

    {
        if (vm.count("pika:queuing"))
            queuing = vm["pika:queuing"].as<std::string>();

        if (vm.count("pika:numa-sensitive"))
            numa_sensitive = 1;
        else
            numa_sensitive = 0;

        const int repetitions = vm["repetitions"].as<int>();

        if (vm.count("info"))
            info_string = vm["info"].as<std::string>();

        num_threads = pika::get_num_worker_threads();

        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const std::uint64_t count = vm["futures"].as<std::uint64_t>();
        bool csv = vm.count("csv") != 0;
        if (PIKA_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        cds::gc::HP hpGC;

        pika::execution::parallel_executor par;
        pika::parallel::execution::parallel_executor_aggregated par_agg;

        for (int i = 0; i < repetitions; i++)
        {
            for (int cds = 0; cds < 2; cds++)
            {
                measure_function_futures_create_thread_hierarchical_placement(
                    count, csv, bool(cds));
            }
        }
    }

    return pika::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("futures", value<std::uint64_t>()->default_value(500000),
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
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(argc, argv, init_args);
}
