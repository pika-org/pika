//  Copyright (c) 2018-2020 Mikael Simberg
//  Copyright (c) 2018-2019 John Biddiscombe
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/local/chrono.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

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

void measure_function_futures_create_thread_hierarchical_placement(
    std::uint64_t count, const int repetitions)
{
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
    auto const desc = pika::util::thread_description();
    auto prio = pika::threads::thread_priority::normal;
    auto const stack_size = pika::threads::thread_stacksize::small_;
    auto const num_threads = pika::get_num_worker_threads();
    pika::error_code ec;

    pika::util::perftests_report(
        "future overhead - create_thread_hierarchical - latch", "no-executor",
        repetitions, [&]() -> void {
            pika::lcos::local::latch l(count);

            auto const func = [&l]() {
                null_function();
                l.count_down(1);
            };
            auto const thread_func =
                pika::threads::detail::thread_function_nullary<decltype(func)>{
                    func};
            for (std::size_t t = 0; t < num_threads; ++t)
            {
                auto const hint = pika::threads::thread_schedule_hint(
                    static_cast<std::int16_t>(t));
                auto spawn_func = [&thread_func, sched, hint, t, count,
                                      num_threads, desc, prio]() {
                    std::uint64_t const count_start = t * count / num_threads;
                    std::uint64_t const count_end =
                        (t + 1) * count / num_threads;
                    pika::error_code ec;
                    for (std::uint64_t i = count_start; i < count_end; ++i)
                    {
                        pika::threads::thread_init_data init(
                            pika::threads::thread_function_type(thread_func),
                            desc, prio, hint, stack_size,
                            pika::threads::thread_schedule_state::pending, false,
                            sched);
                        sched->create_thread(init, nullptr, ec);
                    }
                };
                auto const thread_spawn_func =
                    pika::threads::detail::thread_function_nullary<decltype(
                        spawn_func)>{spawn_func};

                pika::threads::thread_init_data init(
                    pika::threads::thread_function_type(thread_spawn_func), desc,
                    prio, hint, stack_size,
                    pika::threads::thread_schedule_state::pending, false, sched);
                sched->create_thread(init, nullptr, ec);
            }
            l.wait();
        });
    pika::util::perftests_print_times();
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
        if (PIKA_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        if (test_all)
        {
            measure_function_futures_create_thread_hierarchical_placement(
                count, repetitions);
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
