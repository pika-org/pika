//  Copyright (C) 2012-2017 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <array>
#include <string>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

void test_wait_for_either_of_two_futures_list()
{
    std::array<pika::future<int>, 2> futures;
    pika::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures[0] = pt1.get_future();
    pika::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures[1] = pt2.get_future();

    pt1();

    pika::future<pika::when_some_result<std::array<pika::future<int>, 2>>> r =
        pika::when_some(1u, futures);
    pika::when_some_result<std::array<pika::future<int>, 2>> raw = r.get();

    PIKA_TEST_EQ(raw.indices.size(), 1u);
    PIKA_TEST_EQ(raw.indices[0], 0u);

    std::array<pika::future<int>, 2> t = std::move(raw.futures);

    PIKA_TEST(!futures.front().valid());
    PIKA_TEST(!futures.back().valid());

    PIKA_TEST(t.front().is_ready());
    PIKA_TEST_EQ(t.front().get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
using pika::program_options::options_description;
using pika::program_options::variables_map;

using pika::future;

int pika_main(variables_map&)
{
    test_wait_for_either_of_two_futures_list();

    pika::local::finalize();
    return pika::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " PIKA_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
