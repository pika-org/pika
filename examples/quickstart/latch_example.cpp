//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Demonstrate the use of pika::latch

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/latch.hpp>

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
std::ptrdiff_t num_threads = 16;

///////////////////////////////////////////////////////////////////////////////
void wait_for_latch(pika::latch& l) { l.arrive_and_wait(); }

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    num_threads = vm["num-threads"].as<std::ptrdiff_t>();

    pika::latch l(num_threads + 1);

    std::vector<ex::unique_any_sender<>> results;
    for (std::ptrdiff_t i = 0; i != num_threads; ++i)
    {
        results.emplace_back(ex::transfer_just(ex::thread_pool_scheduler{}, std::ref(l)) |
            ex::then(wait_for_latch) | ex::ensure_started());
    }

    // Wait for all threads to reach this point.
    l.arrive_and_wait();

    tt::sync_wait(ex::when_all_vector(std::move(results)));

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    using pika::program_options::options_description;
    using pika::program_options::value;

    // Configure application-specific options
    options_description desc_commandline("Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("num-threads,n", value<std::ptrdiff_t>()->default_value(16),
        "number of threads to synchronize at a local latch (default: 16)");

    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return pika::init(pika_main, argc, argv, init_args);
}
