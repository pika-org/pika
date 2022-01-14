//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>

#include <functional>
#include <iostream>
#include <vector>

int const num_threads = 10;

///////////////////////////////////////////////////////////////////////////////
void wait_for_latch(pika::lcos::local::latch& l)
{
    l.count_down_and_wait();
}

int pika_main()
{
    // Spawn a couple of threads
    pika::lcos::local::latch l(num_threads + 1);

    std::vector<pika::future<void>> results;
    results.reserve(num_threads);

    for (int i = 0; i != num_threads; ++i)
        results.push_back(pika::async(&wait_for_latch, std::ref(l)));

    // Allow spawned threads to reach latch
    pika::this_thread::yield();

    // Enumerate all suspended threads
    pika::threads::enumerate_threads(
        [](pika::threads::thread_id_type id) -> bool {
            std::cout << "thread " << pika::thread::id(id) << " is "
                      << pika::threads::get_thread_state_name(
                             pika::threads::get_thread_state(id))
                      << std::endl;
            return true;    // always continue enumeration
        },
        pika::threads::thread_schedule_state::suspended);

    // Wait for all threads to reach this point.
    l.count_down_and_wait();

    pika::wait_all(results);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
