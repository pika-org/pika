//  Copyright (c) 2015 Martin Stumpf
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1422: pika:threads=all
// allocates too many os threads

#include <pika/local/init.hpp>
#include <pika/local/runtime.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

unsigned long num_cores = 0;

int pika_main()
{
    std::size_t const os_threads = pika::get_os_thread_count();

    std::cout << "Cores: " << num_cores << std::endl;
    std::cout << "OS Threads: " << os_threads << std::endl;

    PIKA_TEST_EQ(num_cores, os_threads);

    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    // Get number of cores from OS
    num_cores = pika::threads::hardware_concurrency();

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");
    return pika::util::report_errors();
}
