//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
# include <pika/future.hpp>
# include <pika/init.hpp>
# include <pika/modules/program_options.hpp>
# include <pika/modules/timing.hpp>

# include <fmt/ostream.h>
# include <fmt/printf.h>

# include <cstddef>
# include <cstdint>
# include <iostream>
# include <numeric>
# include <vector>

# include "worker_timed.hpp"

///////////////////////////////////////////////////////////////////////////////
std::size_t iterations = 10000;
std::uint64_t delay = 0;

void just_wait()
{
    worker_timed(delay * 1000);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
double measure_one(Policy policy)
{
    std::vector<pika::future<void>> threads;
    threads.reserve(iterations);

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i != iterations; ++i)
    {
        threads.push_back(pika::async(policy, &just_wait));
    }

    pika::wait_all(threads);

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

template <typename Policy>
double measure(Policy policy)
{
    std::size_t num_cores = pika::get_os_thread_count();
    std::vector<pika::future<double>> cores;
    cores.reserve(num_cores);

    for (std::size_t i = 0; i != num_cores; ++i)
    {
        cores.push_back(pika::async(&measure_one<Policy>, policy));
    }

    std::vector<double> times = pika::unwrap(cores);
    return std::accumulate(times.begin(), times.end(), 0.0);
}

int pika_main(pika::program_options::variables_map& vm)
{
    bool print_header = vm.count("no-header") == 0;
    bool do_child = vm.count("no-child") == 0;      // fork only
    bool do_parent = vm.count("no-parent") == 0;    // async only
    std::size_t num_cores = pika::get_os_thread_count();
    if (vm.count("num_cores") != 0)
        num_cores = vm["num_cores"].as<std::size_t>();

    // first collect child stealing times
    double child_stealing_time = 0;
    if (do_parent)
        child_stealing_time = measure(pika::launch::async);

    // now collect parent stealing times
    double parent_stealing_time = 0;
    if (do_child)
        parent_stealing_time = measure(pika::launch::fork);

    if (print_header)
    {
        std::cout << "num_cores,num_threads,child_stealing_time[s],parent_"
                     "stealing_time[s]"
                  << std::endl;
    }

    fmt::print(std::cout, "{},{},{},{}\n", num_cores, iterations, child_stealing_time,
        parent_stealing_time);

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    namespace po = pika::program_options;
    po::options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("delay",
            po::value<std::uint64_t>(&delay)->default_value(0),
            "time to busy wait in delay loop [microseconds] "
            "(default: no busy waiting)")
        ("num_threads",
            po::value<std::size_t>(&iterations)->default_value(10000),
            "number of threads to create while measuring execution "
            "(default: 10000)")
        ("num_cores",
            po::value<std::size_t>(),
            "number of spawning tasks to execute (default: number of cores)")
        ("no-header", "do not print out the csv header row")
        ("no-child", "do not test child-stealing (launch::fork only)")
        ("no-parent", "do not test child-stealing (launch::async only)")
        ;
    // clang-format on

    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
#endif
