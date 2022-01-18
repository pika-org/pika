//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/timing.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::vector<pika::future<void>> create_tasks(
    std::size_t num_tasks, std::size_t delay)
{
    std::vector<pika::future<void>> tasks;
    tasks.reserve(num_tasks);
    for (std::size_t i = 0; i != num_tasks; ++i)
    {
        if (delay == 0)
        {
            tasks.push_back(pika::make_ready_future());
        }
        else
        {
            tasks.push_back(
                pika::make_ready_future_after(std::chrono::microseconds(delay)));
        }
    }
    return tasks;
}

double wait_tasks(std::size_t num_samples, std::size_t num_tasks,
    std::size_t num_chunks, std::size_t delay)
{
    std::size_t num_chunk_tasks = ((num_tasks + num_chunks) / num_chunks) - 1;
    std::size_t last_num_chunk_tasks =
        num_tasks - (num_chunks - 1) * num_chunk_tasks;

    double result = 0;

    for (std::size_t k = 0; k != num_samples; ++k)
    {
        std::vector<std::vector<pika::future<void>>> chunks;
        chunks.reserve(num_chunks);
        for (std::size_t c = 0; c != num_chunks - 1; ++c)
        {
            chunks.push_back(create_tasks(num_chunk_tasks, delay));
        }
        chunks.push_back(create_tasks(last_num_chunk_tasks, delay));

        std::vector<pika::future<void>> chunk_results;
        chunk_results.reserve(num_chunks);

        // wait of tasks in chunks
        pika::chrono::high_resolution_timer t;
        if (num_chunks == 1)
        {
            pika::wait_all(chunks[0]);
        }
        else
        {
            for (std::size_t c = 0; c != num_chunks; ++c)
            {
                chunk_results.push_back(
                    pika::async([&chunks, c]() { pika::wait_all(chunks[c]); }));
            }
            pika::wait_all(chunk_results);
        }
        result += t.elapsed();
    }

    return result / num_samples;
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    std::size_t num_samples = 1000;
    std::size_t num_tasks = 100;
    std::size_t num_chunks = 1;
    std::size_t delay = 0;
    bool header = true;

    if (vm.count("no-header"))
        header = false;
    if (vm.count("samples"))
        num_samples = vm["samples"].as<std::size_t>();
    if (vm.count("futures"))
        num_tasks = vm["futures"].as<std::size_t>();
    if (vm.count("chunks"))
        num_chunks = vm["chunks"].as<std::size_t>();
    if (vm.count("delay"))
        delay = vm["delay"].as<std::size_t>();

    if (num_chunks == 0)
        num_chunks = 1;

    // wait for all of the tasks sequentially
    double elapsed_seq = wait_tasks(num_samples, num_tasks, 1, delay);

    // wait of tasks in chunks
    double elapsed_chunks = 0;
    if (num_chunks != 1)
        elapsed_chunks = wait_tasks(num_samples, num_tasks, num_chunks, delay);

    if (header)
    {
        std::cout
            << "Tasks,Chunks,Delay[s],Total Walltime[s],Walltime per Task[s]"
            << std::endl;
    }

    std::string const tasks_str = pika::util::format("{}", num_tasks);
    std::string const chunks_str = pika::util::format("{}", num_chunks);
    std::string const delay_str = pika::util::format("{}", delay);

    pika::util::format_to(std::cout, "{:10},{:10},{:10},{:10},{:10.12}\n",
        tasks_str, std::string("1"), delay_str, elapsed_seq,
        elapsed_seq / num_tasks)
        << std::endl;
    pika::util::print_cdash_timing("WaitAll", elapsed_seq / num_tasks);

    if (num_chunks != 1)
    {
        pika::util::format_to(std::cout,
            "{:10},{:10},{:10},{:10},{:10.12},{:10.12}\n", tasks_str,
            chunks_str, delay_str, elapsed_chunks, elapsed_chunks / num_tasks)
            << std::endl;
        pika::util::print_cdash_timing(
            "WaitAllChunks", elapsed_chunks / num_tasks);
    }
    return pika::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    namespace po = pika::program_options;

    // Configure application-specific options.
    po::options_description cmdline(
        "usage: " PIKA_APPLICATION_STRING " [options]");
    cmdline.add_options()("samples,s",
        po::value<std::size_t>()->default_value(1000),
        "number of tasks to concurrently wait for (default: 1000)")("futures,f",
        po::value<std::size_t>()->default_value(100),
        "number of tasks to concurrently wait for (default: 100)") ("chunks,c",
        po::value<std::size_t>()->default_value(1),
        "number of chunks to split tasks into (default: 1)") ("delay,d",
        po::value<std::size_t>()->default_value(0),
        "number of iterations in the delay loop") ("no-header,n",
        po::value<bool>()->default_value(true),
        "do not print out the csv header row");

    // Initialize and run pika.
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::local::init(pika_main, argc, argv, init_args);
}
#endif
