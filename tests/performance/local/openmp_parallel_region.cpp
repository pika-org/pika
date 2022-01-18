//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to enter and exit an OpenMP
// parallel region. This is meant to be compared to resume_suspend and
// start_stop.

#include <pika/execution_base/this_thread.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/modules/timing.hpp>
#include <pika/type_support/unused.hpp>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

int main(int argc, char** argv)
{
    pika::program_options::options_description desc_commandline;
    desc_commandline.add_options()("repetitions",
        pika::program_options::value<std::uint64_t>()->default_value(100),
        "Number of repetitions");

    pika::program_options::variables_map vm;
    pika::program_options::store(
        pika::program_options::command_line_parser(argc, argv)
            .allow_unregistered()
            .options(desc_commandline)
            .run(),
        vm);

    std::uint64_t repetitions = vm["repetitions"].as<std::uint64_t>();

    // Do one warmup iteration and get the number of threads
    int x = 0;
#pragma omp parallel
    {
        x += 1;
    }
    PIKA_UNUSED(x);

    std::size_t threads = omp_get_max_threads();

    std::cout << "threads, parallel region [s]" << std::endl;

    pika::chrono::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        // TODO: Is there a more minimal way of starting all OpenMP threads?
#pragma omp parallel
        {
            x += 1;
        }

        auto t_parallel = timer.elapsed();

        std::cout << threads << ", " << t_parallel << std::endl;
    }
}
