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
#include <chrono>
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

void test_wait_for_all_from_array()
{
    unsigned const count = 10;
    std::array<pika::future<int>, 10> futures;
    for (unsigned j = 0; j < count; ++j)
    {
        pika::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures[j] = task.get_future();
        task.apply();
    }

    pika::future<std::array<pika::future<int>, 10>> r = pika::when_all(futures);

    std::array<pika::future<int>, 10> result = r.get();

    for (const auto& f : futures)
        PIKA_TEST(!f.valid());
    for (const auto& r : result)
        PIKA_TEST(r.is_ready());
}

///////////////////////////////////////////////////////////////////////////////
using pika::program_options::options_description;
using pika::program_options::variables_map;

using pika::future;

int pika_main(variables_map&)
{
    test_wait_for_all_from_array();

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
