// Copyright (C) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

#define NUM_YIELD_TESTS 1000

///////////////////////////////////////////////////////////////////////////////
void test_yield()
{
    for (std::size_t i = 0; i != NUM_YIELD_TESTS; ++i) pika::this_thread::yield();
}

int pika_main()
{
    auto sched = ex::thread_pool_scheduler{};

    std::size_t num_cores = pika::get_os_thread_count();

    std::vector<ex::unique_any_sender<>> finished;
    finished.reserve(num_cores);

    for (std::size_t i = 0; i != num_cores; ++i)
    {
        finished.emplace_back(ex::schedule(sched) | ex::then(test_yield));
    }

    tt::sync_wait(ex::when_all_vector(std::move(finished)));

    return pika::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    pika::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ(pika::init(pika_main, argc, argv, init_args), 0);
    return 0;
}
