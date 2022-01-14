//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example benchmarks the time it takes to resume and suspend the pika
// runtime. This is meant to be compared to start_stop and
// openmp_parallel_region.

#include <pika/execution_base/this_thread.hpp>
#include <pika/local/chrono.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <pika/modules/program_options.hpp>

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

    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    pika::local::start(nullptr, argc, argv, init_args);
    pika::local::suspend();

    std::uint64_t threads = pika::resource::get_num_threads("default");

    std::cout << "threads, resume [s], apply [s], suspend [s]" << std::endl;

    double suspend_time = 0;
    double resume_time = 0;
    pika::chrono::high_resolution_timer timer;

    for (std::size_t i = 0; i < repetitions; ++i)
    {
        timer.restart();

        pika::local::resume();
        auto t_resume = timer.elapsed();
        resume_time += t_resume;

        for (std::size_t thread = 0; thread < threads; ++thread)
        {
            pika::apply([]() {});
        }

        auto t_apply = timer.elapsed();

        pika::local::suspend();
        auto t_suspend = timer.elapsed();
        suspend_time += t_suspend;

        std::cout << threads << ", " << t_resume << ", " << t_apply << ", "
                  << t_suspend << std::endl;
    }

    pika::util::print_cdash_timing("ResumeTime", resume_time);
    pika::util::print_cdash_timing("SuspendTime", suspend_time);

    pika::local::resume();
    pika::apply([]() { pika::local::finalize(); });
    pika::local::stop();
}
