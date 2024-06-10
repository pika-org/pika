//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/timing.hpp>
#include <pika/runtime.hpp>
#include <pika/testing/performance.hpp>
#include <pika/thread.hpp>

#include <fmt/format.h>
#include <fmt/printf.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace po = pika::program_options;

//#define DEBUG(x) std::cout << x << std::endl;
#define DEBUG(x)

// ------------------------------------------------
struct task_data
{
    pika::detail::spinlock mtx_;
    pika::condition_variable cv_;
    bool ping_ready_{false};
    bool pong_ready_{false};
};

// ------------------------------------------------
void function_A(std::uint64_t loops, task_data* tda, task_data* tdb)
{
    for (std::uint64_t i = 0; i < loops; ++i)
    {
        // -------------------
        // PING
        // -------------------
        // sleep until thread B notifies us
        {
            std::unique_lock lk1(tda->mtx_);
            tda->cv_.wait(lk1, [tda] { return tda->ping_ready_; });
        }
        DEBUG("A : step 1 complete");

        // notify thread B that we are ready
        {
            std::lock_guard lk1(tdb->mtx_);
            tdb->pong_ready_ = false;
            tdb->ping_ready_ = true;
        }
        tdb->cv_.notify_one();
        DEBUG("A : step 2 complete");

        // -------------------
        // PONG
        // -------------------
        // sleep until thread B notifies us
        {
            std::unique_lock lk2(tda->mtx_);
            tda->cv_.wait(lk2, [tda] { return tda->pong_ready_; });
        }
        DEBUG("A : step 3 complete");

        // notify thread B that we are ready
        {
            std::lock_guard lk2(tdb->mtx_);
            tdb->ping_ready_ = false;
            tdb->pong_ready_ = true;
        }
        // wake up thread B
        tdb->cv_.notify_one();
        DEBUG("A : step 4 complete");
    }
}

// ------------------------------------------------
void function_B(std::uint64_t loops, task_data* tda, task_data* tdb)
{
    for (std::uint64_t i = 0; i < loops; ++i)
    {
        // -------------------
        // PING
        // -------------------
        // notify thread A that we are ready
        {
            std::lock_guard lk1(tda->mtx_);
            tda->pong_ready_ = false;
            tda->ping_ready_ = true;
        }
        tda->cv_.notify_one();
        DEBUG("B : step 1 complete");

        // sleep until thread A wakes us up
        {
            std::unique_lock lk1(tdb->mtx_);
            tdb->cv_.wait(lk1, [tdb] { return tdb->ping_ready_; });
        }
        DEBUG("B : step 2 complete");

        // -------------------
        // PONG
        // -------------------
        // notify thread A that we are ready
        {
            std::lock_guard lk2(tda->mtx_);
            tda->ping_ready_ = false;
            tda->pong_ready_ = true;
        }
        tda->cv_.notify_one();
        DEBUG("B : step 3 complete");

        // sleep until thread A wakes us up
        {
            std::unique_lock lk2(tdb->mtx_);
            tdb->cv_.wait(lk2, [tdb] { return tdb->pong_ready_; });
        }
        DEBUG("B : step 4 complete");
    }
}

// ------------------------------------------------
template <typename Scheduler>
void test_cv(Scheduler&& sched, std::uint64_t loops)
{
    int N = pika::get_num_worker_threads() / 2;
    //
    std::vector<ex::unique_any_sender<>> senders;
    std::vector<task_data> task_data_A(N);
    std::vector<task_data> task_data_B(N);
    //
    for (int i = 0; i < N; i++)
    {
        // thread A
        auto s1 = ex::transfer_just(sched, &task_data_A[i], &task_data_B[i])    //
            | ex::then([&, loops](task_data* tda, task_data* tdb) {             //
                  function_A(loops, tda, tdb);
              });
        senders.push_back(std::move(s1));

        // thread B
        auto s2 = ex::transfer_just(sched, &task_data_A[i], &task_data_B[i])    //
            | ex::then([&, loops](task_data* tda, task_data* tdb) {             //
                  function_B(loops, tda, tdb);
              });
        senders.push_back(std::move(s2));
    }

    pika::this_thread::experimental::sync_wait(ex::when_all_vector(std::move(senders)));
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(po::variables_map& vm)
{
    using pika::chrono::detail::high_resolution_timer;

    auto const loops = vm["loops"].as<std::uint64_t>();
    auto const repetitions = vm["repetitions"].as<std::uint64_t>();
    auto const perftest_json = vm["perftest-json"].as<bool>();

    double time_avg_s = 0.0;
    double time_min_s = std::numeric_limits<double>::max();
    double time_max_s = std::numeric_limits<double>::min();

    // use a small stack for this test
    auto sched =
        ex::with_stacksize(ex::thread_pool_scheduler(), pika::execution::thread_stacksize::minimal);

    for (std::uint64_t i = 0; i < repetitions; i++)
    {
        high_resolution_timer timer;
        test_cv(sched, loops);
        double time_s = timer.elapsed();
        time_avg_s += time_s;
        time_max_s = (std::max)(time_max_s, time_s);
        time_min_s = (std::min)(time_min_s, time_s);
        if (!perftest_json) { fmt::print("iteration {}, time {}\n", i, time_s); }
    }

    time_avg_s /= repetitions;

    // use 0.5 as we use 2 wait+notifies on each loop
    double const time_avg_us = 0.5 * time_avg_s * 1e6 / loops;
    double const time_min_us = 0.5 * time_min_s * 1e6 / loops;
    double const time_max_us = 0.5 * time_max_s * 1e6 / loops;

    if (perftest_json)
    {
        pika::util::detail::json_perf_times t;
        t.add(
            fmt::format("condition_variable_overhead - {} threads", pika::get_num_worker_threads()),
            time_avg_us);
        std::cout << t;
    }
    else
    {
        fmt::print("repetitions, time_avg_us, time_min_us, time_max_us\n");
        fmt::print(
            "{}, {:9.2f}, {:9.2f}, {:9.2f}\n", repetitions, time_avg_us, time_min_us, time_max_us);
    }

    pika::finalize();
    return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    po::options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("loops", po::value<std::uint64_t>()->default_value(10000), "number of loops inside the test function")
        ("repetitions", po::value<std::uint64_t>()->default_value(10), "number of repetitions of the benchmark")
        ("perftest-json", po::bool_switch(), "print final task size in json format for use with performance CI")
        // clang-format on
        ;

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
