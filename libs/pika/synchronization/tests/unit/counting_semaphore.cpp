//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/semaphore.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> count(0);

void worker(pika::lcos::local::counting_semaphore& sem)
{
    ++count;
    sem.signal();    // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    pika::lcos::local::counting_semaphore sem;

    for (std::size_t i = 0; i != 10; ++i)
        pika::apply(&worker, std::ref(sem));

    // Wait for all threads to finish executing.
    sem.wait(10);

    PIKA_TEST_EQ(count, 10);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
