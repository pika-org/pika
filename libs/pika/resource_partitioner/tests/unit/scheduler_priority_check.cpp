//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that creates a set of tasks using normal priority, but every
// Nth normal task spawns a set of high priority tasks.
// The test is intended to be used with a task plotting/profiling
// tool to verify that high priority tasks run before low ones.

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>
#include <pika/threading_base/annotated_function.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

// dummy function we will call using execute
void dummy_task(std::size_t n)
{
    // no other work can take place on this thread whilst it sleeps
    bool sleep = true;
    auto start = std::chrono::steady_clock::now();
    do {
        std::this_thread::sleep_for(std::chrono::microseconds(n) / 25);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
        sleep = (elapsed < std::chrono::microseconds(n));
    } while (sleep);
}

inline std::size_t st_rand() { return std::size_t(std::rand()); }

int pika_main(variables_map& vm)
{
    auto const sched = pika::threads::detail::get_self_id_data()->get_scheduler_base();
    std::cout << "Scheduler is " << sched->get_description() << std::endl;
    if (std::string("core-shared_priority_queue_scheduler") == sched->get_description())
    {
        using ::pika::threads::scheduler_mode;
        std::cout << "Setting shared-priority mode flags" << std::endl;
        sched->add_scheduler_mode(scheduler_mode::enable_stealing |
            scheduler_mode::enable_stealing_numa | scheduler_mode::assign_work_round_robin |
            scheduler_mode::steal_high_priority_first);
        sched->remove_scheduler_mode(scheduler_mode::assign_work_thread_parent |
            scheduler_mode::steal_after_local | scheduler_mode::reduce_thread_priority |
            scheduler_mode::enable_elasticity);
    }

    // setup schedulers for different task priorities on the pools
    auto HP_scheduler =
        ex::with_priority(ex::thread_pool_scheduler{&pika::resource::get_thread_pool("default")},
            pika::execution::thread_priority::high);

    auto NP_scheduler =
        ex::with_priority(ex::thread_pool_scheduler{&pika::resource::get_thread_pool("default")},
            pika::execution::thread_priority::default_);

    // randomly create normal priority tasks
    // and then a set of HP tasks in periodic bursts
    // Use task plotting tools to validate that scheduling is correct
    const int np_loop = vm["nnp"].as<int>();
    const int hp_loop = vm["nhp"].as<int>();
    const int np_m = vm["mnp"].as<int>();
    const int hp_m = vm["mhp"].as<int>();
    const int cycles = vm["cycles"].as<int>();

    const int np_total = np_loop * cycles;
    //
    struct dec_counter
    {
        explicit dec_counter(std::atomic<int>& counter)
          : counter_(counter)
        {
        }
        ~dec_counter() { --counter_; }
        //
        std::atomic<int>& counter_;
    };

    // diagnostic counters for debugging profiler numbers
    std::atomic<int> np_task_count(0);
    std::atomic<int> hp_task_count(0);
    std::atomic<int> hp_launch_count(0);
    std::atomic<int> launch_count(0);
    //
    std::atomic<int> count_down((np_loop + hp_loop) * cycles);
    std::atomic<int> counter(0);
    ex::execute(NP_scheduler,
        pika::annotated_function(
            [&]() {
                ++launch_count;
                for (int i = 0; i < np_total; ++i)
                {
                    // normal priority
                    auto s = ex::schedule(NP_scheduler) |
                        ex::then(pika::annotated_function(
                            [&, np_m]() {
                                np_task_count++;
                                dec_counter dec(count_down);
                                dummy_task(std::size_t(np_m));
                            },
                            "NP task")) |
                        // continuation runs as a sync task
                        ex::then([&]() {
                            // on every Nth task, spawn new HP tasks, otherwise quit
                            if ((++counter) % np_loop != 0) return;

                            // Launch HP tasks using an HP task to do it
                            ex::execute(HP_scheduler,
                                pika::annotated_function(
                                    [&]() {
                                        ++hp_launch_count;
                                        for (int j = 0; j < hp_loop; ++j)
                                        {
                                            ex::execute(HP_scheduler,
                                                pika::annotated_function(
                                                    [&]() {
                                                        ++hp_task_count;
                                                        dec_counter dec(count_down);
                                                        dummy_task(std::size_t(hp_m));
                                                    },
                                                    "HP task"));
                                        }
                                    },
                                    "Launch HP"));
                        });
                    ex::start_detached(std::move(s));
                }
            },
            "Launch"));

    // wait for everything to finish
    do {
        pika::this_thread::yield();
    } while (count_down > 0);

    std::cout << "Tasks NP  : " << np_task_count << "\n"
              << "Tasks HP  : " << hp_task_count << "\n"
              << "Launch    : " << launch_count << "\n"
              << "Launch HP : " << hp_launch_count << std::endl;

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("nnp", value<int>()->default_value(100),
         "number of Normal Priority futures per cycle")

        ("nhp", value<int>()->default_value(50),
         "number of High Priority futures per cycle")

        ("mnp", value<int>()->default_value(1000),
         "microseconds per Normal Priority task")

        ("mhp", value<int>()->default_value(100),
         "microseconds per High Priority task")

        ("cycles", value<int>()->default_value(10),
         "number of cycles in the benchmark");
    // clang-format on

    // Setup the init parameters
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    PIKA_TEST_EQ(pika::init(pika_main, argc, argv, init_args), 0);

    return 0;
}
