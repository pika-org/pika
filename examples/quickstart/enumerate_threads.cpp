//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/thread.hpp>

#include <functional>
#include <iostream>
#include <vector>

int const num_threads = 10;

///////////////////////////////////////////////////////////////////////////////
void wait_for_latch(pika::latch& l)
{
    l.arrive_and_wait();
}

int pika_main()
{
    // Spawn a couple of threads
    pika::latch l(num_threads + 1);

    std::vector<pika::future<void>> results;
    results.reserve(num_threads);

    for (int i = 0; i != num_threads; ++i)
        results.push_back(pika::async(&wait_for_latch, std::ref(l)));

    // Allow spawned threads to reach latch
    pika::this_thread::yield();

    // Enumerate all suspended threads
    pika::threads::enumerate_threads(
        [](pika::threads::detail::thread_id_type id) -> bool {
            std::cout << "thread " << pika::thread::id(id) << " is "
                      << pika::threads::detail::get_thread_state_name(
                             pika::threads::detail::get_thread_state(id))
                      << std::endl;
            return true;    // always continue enumeration
        },
        pika::threads::detail::thread_schedule_state::suspended);

    // Wait for all threads to reach this point.
    l.arrive_and_wait();

    pika::wait_all(results);

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    return pika::init(pika_main, argc, argv);
}
