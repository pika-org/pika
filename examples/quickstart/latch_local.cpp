//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Demonstrate the use of pika::lcos::local::latch

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/latch.hpp>

#include <cstddef>
#include <functional>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::ptrdiff_t num_threads = 16;

///////////////////////////////////////////////////////////////////////////////
void wait_for_latch(pika::lcos::local::latch& l)
{
    l.count_down_and_wait();
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    num_threads = vm["num-threads"].as<std::ptrdiff_t>();

    pika::lcos::local::latch l(num_threads + 1);

    std::vector<pika::future<void>> results;
    for (std::ptrdiff_t i = 0; i != num_threads; ++i)
        results.push_back(pika::async(&wait_for_latch, std::ref(l)));

    // Wait for all threads to reach this point.
    l.count_down_and_wait();

    pika::wait_all(results);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    using pika::program_options::options_description;
    using pika::program_options::value;

    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("num-threads,n",
        value<std::ptrdiff_t>()->default_value(16),
        "number of threads to synchronize at a local latch (default: 16)");

    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return pika::local::init(pika_main, argc, argv, init_args);
}
