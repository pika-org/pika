//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how to customize certain
// parameters (such like thread priority, the stacksize, or the targeted
// processing unit) for a thread which is created with schedule/transfer.

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/thread.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
void run_with_large_stack()
{
    int const array_size = 1000000;

    // Allocating a huge array on the stack would normally cause problems.
    // For this reason, this function is scheduled on a thread using a large
    // stack (see below).
    char large_array[array_size];    // allocate 1 MByte of memory

    std::fill(large_array, &large_array[array_size], '\0');

    std::cout << "This thread runs with a "
              << pika::detail::threads::get_stack_size_name(pika::this_thread::get_stack_size())
              << " stack and "
              << pika::execution::detail::get_thread_priority_name(
                     pika::this_thread::get_priority())
              << " priority." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_with_high_priority()
{
    std::cout << "This thread runs with "
              << pika::execution::detail::get_thread_priority_name(
                     pika::this_thread::get_priority())
              << " priority." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    // run a thread on a large stack
    {
        auto large_stack_scheduler = ex::with_stacksize(
            ex::thread_pool_scheduler{}, pika::execution::thread_stacksize::large);

        tt::sync_wait(ex::schedule(large_stack_scheduler) | ex::then(run_with_large_stack));
    }

    // run a thread with high priority
    {
        auto high_priority_scheduler =
            ex::with_priority(ex::thread_pool_scheduler{}, pika::execution::thread_priority::high);

        tt::sync_wait(ex::schedule(high_priority_scheduler) | ex::then(run_with_high_priority));
    }

    // combine both
    {
        auto fancy_scheduler = ex::with_stacksize(
            ex::with_priority(ex::thread_pool_scheduler{}, pika::execution::thread_priority::high),
            pika::execution::thread_stacksize::large);

        tt::sync_wait(ex::schedule(fancy_scheduler) | ex::then(run_with_large_stack));
    }

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
