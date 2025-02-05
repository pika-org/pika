//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to start and stop the pika runtime.
// This is meant to be compared to resume_suspend and openmp_parallel_region.

#include <pika/chrono.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/testing/performance.hpp>
#include <pika/thread.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace ex = pika::execution::experimental;

int pika_main()
{
    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    using namespace pika::program_options;
    options_description desc_commandline;
    // clang-format off
    desc_commandline.add_options()
        ("repetitions", value<std::uint64_t>()->default_value(100), "Number of repetitions")
        ("perftest-json", bool_switch(), "Print average start-stop time in json format for use with performance CI");
    // clang-format on

    variables_map vm;
    store(command_line_parser(argc, argv).allow_unregistered().options(desc_commandline).run(), vm);

    auto const repetitions = vm["repetitions"].as<std::uint64_t>();
    auto const perftest_json = vm["perftest-json"].as<bool>();

    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    pika::start(pika_main, argc, argv, init_args);
    std::uint64_t threads = pika::resource::get_num_threads("default");
    pika::stop();

    if (!perftest_json) { std::cout << "threads, start [s], stop [s]" << std::endl; }

    double stop_time = 0;
    pika::chrono::detail::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        pika::init_params init_args;
        init_args.desc_cmdline = desc_commandline;

        pika::start(pika_main, argc, argv, init_args);
        auto t_start = timer.elapsed();

        pika::stop();
        auto t_stop = timer.elapsed();
        stop_time += t_stop;

        if (!perftest_json)
        {
            std::cout << threads << ", " << t_start << ", " << t_stop << std::endl;
        }
    }

    if (perftest_json)
    {
        pika::util::detail::json_perf_times t;
        t.add(fmt::format("start_stop - {} threads", threads), stop_time / repetitions);
        std::cout << t;
    }
}
