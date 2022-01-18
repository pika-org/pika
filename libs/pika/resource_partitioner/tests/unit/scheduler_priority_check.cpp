//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that creates a set of tasks using normal priority, but every
// Nth normal task spawns a set of high priority tasks.
// The test is intended to be used with a task plotting/profiling
// tool to verify that high priority tasks run before low ones.

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>
#include <pika/program_options.hpp>
#include <pika/threading_base/annotated_function.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

// dummy function we will call using async
void dummy_task(std::size_t n)
{
    // no other work can take place on this thread whilst it sleeps
    bool sleep = true;
    auto start = std::chrono::steady_clock::now();
    do
    {
        std::this_thread::sleep_for(std::chrono::microseconds(n) / 25);
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start);
        sleep = (elapsed < std::chrono::microseconds(n));
    } while (sleep);
}

inline std::size_t st_rand()
{
    return std::size_t(std::rand());
}

int pika_main(variables_map& vm)
{
    auto const sched = pika::threads::get_self_id_data()->get_scheduler_base();
    std::cout << "Scheduler is " << sched->get_description() << std::endl;
    if (std::string("core-shared_priority_queue_scheduler") ==
        sched->get_description())
    {
        std::cout << "Setting shared-priority mode flags" << std::endl;
        sched->add_remove_scheduler_mode(
            // add these flags
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::enable_stealing |
                pika::threads::policies::enable_stealing_numa |
                pika::threads::policies::assign_work_round_robin |
                pika::threads::policies::steal_high_priority_first),
            // remove these flags
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::assign_work_thread_parent |
                pika::threads::policies::steal_after_local |
                pika::threads::policies::do_background_work |
                pika::threads::policies::reduce_thread_priority |
                pika::threads::policies::delay_exit |
                pika::threads::policies::fast_idle_mode |
                pika::threads::policies::enable_elasticity));
    }

    // setup executors for different task priorities on the pools
    pika::execution::parallel_executor HP_executor(
        &pika::resource::get_thread_pool("default"),
        pika::threads::thread_priority::high);

    pika::execution::parallel_executor NP_executor(
        &pika::resource::get_thread_pool("default"),
        pika::threads::thread_priority::default_);

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
        ~dec_counter()
        {
            --counter_;
        }
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
    auto f3 = pika::async(NP_executor,
        pika::annotated_function(
            [&]() {
                ++launch_count;
                for (int i = 0; i < np_total; ++i)
                {
                    // normal priority
                    auto f3 = pika::async(NP_executor,
                        pika::annotated_function(
                            [&, np_m]() {
                                np_task_count++;
                                dec_counter dec(count_down);
                                dummy_task(std::size_t(np_m));
                            },
                            "NP task"));

                    // continuation runs as a sync task
                    f3.then(pika::launch::sync, [&](pika::future<void>&&) {
                        // on every Nth task, spawn new HP tasks, otherwise quit
                        if ((++counter) % np_loop != 0)
                            return;

                        // Launch HP tasks using an HP task to do it
                        pika::async(HP_executor,
                            pika::annotated_function(
                                [&]() {
                                    ++hp_launch_count;
                                    for (int j = 0; j < hp_loop; ++j)
                                    {
                                        pika::async(HP_executor,
                                            pika::annotated_function(
                                                [&]() {
                                                    ++hp_task_count;
                                                    dec_counter dec(count_down);
                                                    dummy_task(
                                                        std::size_t(hp_m));
                                                },
                                                "HP task"));
                                    }
                                },
                                "Launch HP"));
                    });
                }
            },
            "Launch"));

    // wait for everything to finish
    do
    {
        pika::this_thread::yield();
    } while (count_down > 0);

    std::cout << "Tasks NP  : " << np_task_count << "\n"
              << "Tasks HP  : " << hp_task_count << "\n"
              << "Launch    : " << launch_count << "\n"
              << "Launch HP : " << hp_launch_count << std::endl;

    return pika::local::finalize();
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
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);

    return pika::util::report_errors();
}
