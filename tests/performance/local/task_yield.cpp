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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>

namespace ex = pika::execution::experimental;
namespace po = pika::program_options;
namespace tt = pika::this_thread::experimental;

template <typename Scheduler>
double test_yield(Scheduler&& sched, std::size_t num_yields)
{
    return tt::sync_wait(ex::schedule(std::forward<Scheduler>(sched)) | ex::then([num_yields]() {
        pika::chrono::detail::high_resolution_timer timer;
        for (std::size_t i = 0; i < num_yields; ++i) { pika::this_thread::yield(); }
        return timer.elapsed();
    }));
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(po::variables_map& vm)
{
    auto const num_yields = vm["num-yields"].as<std::uint64_t>();
    auto const repetitions = vm["repetitions"].as<std::uint64_t>();
    auto const perftest_json = vm["perftest-json"].as<bool>();

    double time_avg_s = 0.0;
    double time_min_s = std::numeric_limits<double>::max();
    double time_max_s = std::numeric_limits<double>::min();

    for (std::uint64_t i = 0; i < repetitions; ++i)
    {
        auto sched = ex::thread_pool_scheduler();
        sched = ex::with_hint(std::move(sched),
            pika::execution::thread_schedule_hint(pika::get_worker_thread_num() + 1));

        double time_s = test_yield(sched, num_yields);

        time_avg_s += time_s;
        time_max_s = (std::max)(time_max_s, time_s);
        time_min_s = (std::min)(time_min_s, time_s);
    }

    time_avg_s /= repetitions;

    double const time_avg_us = time_avg_s * 1e6 / num_yields;
    double const time_min_us = time_min_s * 1e6 / num_yields;
    double const time_max_us = time_max_s * 1e6 / num_yields;

    if (perftest_json)
    {
        pika::util::detail::json_perf_times t;
        t.add(fmt::format("task_yield - {} threads - {}", pika::get_num_worker_threads(),
                  "thread_pool_scheduler"),
            time_avg_us);
        std::cout << t;
    }
    else
    {
        fmt::print("repetitions,time_avg_us,time_min_us,time_max_us\n");
        fmt::print("{},{},{},{}\n", repetitions, time_avg_us, time_min_us, time_max_us);
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
        ("num-yields", po::value<std::uint64_t>()->default_value(100000), "number of yields per repetition")
        ("repetitions", po::value<std::uint64_t>()->default_value(10), "number of repetitions of the benchmark")
        ("perftest-json", po::bool_switch(), "print final task size in json format for use with performance CI")
        // clang-format on
        ;

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
