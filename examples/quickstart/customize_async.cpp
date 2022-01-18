//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how to customize certain
// parameters (such like thread priority, the stacksize, or the targeted
// processing unit) for a thread which is created by calling pika::apply() or
// pika::async().

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>

#include <algorithm>
#include <iostream>

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
              << pika::threads::get_stack_size_name(
                     pika::this_thread::get_stack_size())
              << " stack and "
              << pika::threads::get_thread_priority_name(
                     pika::this_thread::get_priority())
              << " priority." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void run_with_high_priority()
{
    std::cout << "This thread runs with "
              << pika::threads::get_thread_priority_name(
                     pika::this_thread::get_priority())
              << " priority." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    // run a thread on a large stack
    {
        pika::execution::parallel_executor large_stack_executor(
            pika::threads::thread_stacksize::large);

        pika::future<void> f =
            pika::async(large_stack_executor, &run_with_large_stack);
        f.wait();
    }

    // run a thread with high priority
    {
        pika::execution::parallel_executor high_priority_executor(
            pika::threads::thread_priority::high);

        pika::future<void> f =
            pika::async(high_priority_executor, &run_with_high_priority);
        f.wait();
    }

    // combine both
    {
        pika::execution::parallel_executor fancy_executor(
            pika::threads::thread_priority::high,
            pika::threads::thread_stacksize::large);

        pika::future<void> f = pika::async(fancy_executor, &run_with_large_stack);
        f.wait();
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
