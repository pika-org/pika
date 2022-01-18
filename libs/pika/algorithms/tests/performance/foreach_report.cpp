//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/executors/parallel_executor_aggregated.hpp>
#include <pika/local/algorithm.hpp>
#include <pika/local/chrono.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include "foreach_scaling_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    delay = vm["work_delay"].as<int>();
    test_count = vm["test_count"].as<int>();
    chunk_size = vm["chunk_size"].as<int>();
    disable_stealing = vm.count("disable_stealing");
    fast_idle_mode = vm.count("fast_idle_mode");

    // verify that input is within domain of program
    if (test_count == 0 || test_count < 0)
    {
        std::cerr << "test_count cannot be zero or negative...\n" << std::flush;
        pika::local::finalize();
        return -1;
    }
    else if (delay < 0)
    {
        std::cerr << "delay cannot be a negative number...\n" << std::flush;
        pika::local::finalize();
        return -1;
    }

    if (disable_stealing)
    {
        pika::threads::remove_scheduler_mode(
            pika::threads::policies::enable_stealing);
    }
    if (fast_idle_mode)
    {
        pika::threads::add_scheduler_mode(
            pika::threads::policies::fast_idle_mode);
    }

    {
        std::vector<std::size_t> data_representation(vector_size);
        std::iota(std::begin(data_representation),
            std::end(data_representation), gen());

        {
            pika::execution::experimental::scheduler_executor<
                pika::execution::experimental::thread_pool_scheduler>
                exec;
            pika::util::perftests_report("for_each", "scheduler_executor",
                test_count,
                [&]() { measure_parallel_foreach(data_representation, exec); });
        }

        {
            pika::execution::parallel_executor exec;
            pika::util::perftests_report("for_each", "parallel_executor",
                test_count,
                [&]() { measure_parallel_foreach(data_representation, exec); });
        }

        {
            pika::execution::experimental::fork_join_executor exec;
            pika::util::perftests_report("for_each", "fork_join_executor",
                test_count,
                [&]() { measure_parallel_foreach(data_representation, exec); });
        }

        pika::util::perftests_print_times();
    }

    return pika::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using namespace pika::program_options;

    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("vector_size", value<std::size_t>()->default_value(1000),
            "size of vector")
        ("work_delay", value<int>()->default_value(1),
            "loop delay per element in nanoseconds")
        ("test_count", value<int>()->default_value(100),
            "number of tests to be averaged")
        ("chunk_size", value<int>()->default_value(0),
            "number of iterations to combine while parallelization")
        ("disable_stealing", "disable thread stealing")
        ("fast_idle_mode", "enable fast idle mode")
        ;
    // clang-format on

    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = {"pika.os_threads=all"};

    return pika::local::init(pika_main, argc, argv, init_args);
}
#endif
