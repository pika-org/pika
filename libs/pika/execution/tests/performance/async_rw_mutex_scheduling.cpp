//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test measures the performance of accessing values through async_rw_mutex. Accesses are
// scheduled on new tasks to test the performance with concurrency. This means that the benchmark
// includes the overhead of creating new tasks, but it represents a more realistic scenario.

#include <pika/config.hpp>
#include <pika/async_rw_mutex.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/runtime.hpp>
#include <pika/testing/performance.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>

using pika::program_options::bool_switch;
using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::chrono::detail::high_resolution_timer;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

template <typename T>
double test_async_rw_mutex(
    std::uint64_t num_iterations, std::uint64_t num_rw_accesses, std::uint64_t num_ro_accesses)
{
    pika::chrono::detail::high_resolution_timer timer;

    {
        ex::async_rw_mutex<T> m;
        ex::thread_pool_scheduler sched;

        for (std::uint64_t i = 0; i < num_iterations; ++i)
        {
            for (std::uint64_t j = 0; j < num_rw_accesses; ++j)
            {
                ex::start_detached(m.readwrite() | ex::continues_on(sched));
            }

            for (std::uint64_t j = 0; j < num_ro_accesses; ++j)
            {
                ex::start_detached(m.read() | ex::continues_on(sched));
            }
        }

        tt::sync_wait(m.readwrite());
    }

    return timer.elapsed();
}

int pika_main(variables_map& vm)
{
    auto const num_iterations = vm["num-iterations"].as<std::uint64_t>();
    auto const num_rw_accesses = vm["num-rw-accesses"].as<std::uint64_t>();
    auto const num_ro_accesses = vm["num-ro-accesses"].as<std::uint64_t>();
    auto const repetitions = vm["repetitions"].as<std::uint64_t>();
    auto const perftest_json = vm["perftest-json"].as<bool>();

    double time_avg_s = 0.0;
    double time_min_s = std::numeric_limits<double>::max();
    double time_max_s = std::numeric_limits<double>::min();

    for (std::uint64_t i = 0; i < repetitions; ++i)
    {
        double time_s = test_async_rw_mutex<void>(num_iterations, num_rw_accesses, num_ro_accesses);

        time_avg_s += time_s;
        time_max_s = (std::max)(time_max_s, time_s);
        time_min_s = (std::min)(time_min_s, time_s);
    }

    time_avg_s /= repetitions;

    double const time_avg_us = time_avg_s * 1e6 / num_iterations;
    double const time_min_us = time_min_s * 1e6 / num_iterations;
    double const time_max_us = time_max_s * 1e6 / num_iterations;

    if (perftest_json)
    {
        pika::util::detail::json_perf_times t;
        t.add(fmt::format("async_rw_mutex - {} threads - {}:{}", pika::get_num_worker_threads(),
                  num_rw_accesses, num_ro_accesses),
            time_avg_us);
        std::cout << t;
    }
    else
    {
        fmt::print(
            "repetitions,iterations,rw_accesses,ro_accesses,time_avg_us,time_min_us,time_max_us\n");
        fmt::print("{},{},{},{},{},{},{}\n", repetitions, num_iterations, num_rw_accesses,
            num_ro_accesses, time_avg_us, time_min_us, time_max_us);
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
        ("num-iterations", value<std::uint64_t>()->default_value(100), "number of times to cycle through read-write and read-only accesses in one test")
        ("num-rw-accesses", value<std::uint64_t>()->default_value(5), "number of consecutive read-write accesses")
        ("num-ro-accesses", value<std::uint64_t>()->default_value(5), "number of consecutive read-only accesses")
        ("repetitions", value<std::uint64_t>()->default_value(1), "number of repetitions of the full benchmark")
        ("perftest-json", bool_switch(), "print final task size in json format for use with performance CI.")
        // clang-format on
        ;

    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;
    return pika::init(pika_main, argc, argv, init_args);
}
