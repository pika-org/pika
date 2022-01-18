// Copyright (C) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

#define NUM_YIELD_TESTS 1000

///////////////////////////////////////////////////////////////////////////////
void test_yield()
{
    for (std::size_t i = 0; i != NUM_YIELD_TESTS; ++i)
        pika::this_thread::yield();
}

int pika_main()
{
    std::size_t num_cores = pika::get_os_thread_count();

    std::vector<pika::future<void>> finished;
    finished.reserve(num_cores);

    for (std::size_t i = 0; i != num_cores; ++i)
        finished.push_back(pika::async(&test_yield));

    pika::wait_all(finished);

    return pika::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
    return pika::util::report_errors();
}
