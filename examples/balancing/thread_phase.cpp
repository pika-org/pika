//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/barrier.hpp>
#include <pika/functional/bind.hpp>
#include <pika/init.hpp>
#include <pika/mutex.hpp>
#include <pika/thread.hpp>

#include <boost/lockfree/queue.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

using boost::lockfree::queue;

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using std::chrono::milliseconds;

using barrier = pika::barrier<>;
using pika::mutex;

using pika::threads::detail::make_thread_function_nullary;
using pika::threads::detail::register_thread;
using pika::threads::detail::thread_init_data;

using pika::threads::detail::get_self;
using pika::threads::detail::get_self_id;
using pika::threads::detail::get_thread_phase;
using pika::threads::detail::set_thread_state;
using pika::threads::detail::thread_id_ref_type;
using pika::threads::detail::thread_id_type;

using value_type = std::pair<thread_id_type, std::size_t>;
using fifo_type = std::vector<value_type>;

///////////////////////////////////////////////////////////////////////////////
void lock_and_wait(mutex& m, barrier& b0, barrier& b1, value_type& entry, std::size_t /* wait */)
{
    // Wait for all threads in this iteration to be created.
    b0.arrive_and_wait();

    // keep this thread alive while being suspended
    thread_id_ref_type this_ = get_self_id();

    while (true)
    {
        // Try to acquire the mutex.
        std::unique_lock<mutex> l(m, std::try_to_lock);

        if (l.owns_lock())
        {
            entry = value_type(this_.noref(), get_thread_phase(this_.noref()));
            break;
        }

        // Schedule a wakeup.
        set_thread_state(
            this_.noref(), milliseconds(30), pika::threads::detail::thread_schedule_state::pending);

        // Suspend this pika thread.
        pika::this_thread::suspend(pika::threads::detail::thread_schedule_state::suspended);
    }

    // Make pika_main wait for us to finish.
    b1.arrive_and_wait();
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    std::size_t pikathread_count = 0;

    if (vm.count("pikathreads")) pikathread_count = vm["pikathreads"].as<std::size_t>();

    std::size_t mutex_count = 0;

    if (vm.count("mutexes")) mutex_count = vm["mutexes"].as<std::size_t>();
    PIKA_ASSERT(mutex_count != 0);

    std::size_t iterations = 0;

    if (vm.count("iterations")) iterations = vm["iterations"].as<std::size_t>();

    std::size_t wait = 0;

    if (vm.count("wait")) wait = vm["wait"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        std::cout << "iteration: " << i << "\n";

        // Have the fifo preallocate storage.
        fifo_type pikathreads(pikathread_count);

        // Allocate the mutexes.
        std::vector<mutex> m(mutex_count);
        barrier b0(pikathread_count + 1), b1(pikathread_count + 1);

        // keep created threads alive while they are suspended
        std::vector<thread_id_ref_type> ids;
        for (std::size_t j = 0; j < pikathread_count; ++j)
        {
            // Compute the mutex to be used for this thread.
            std::size_t const index = j % mutex_count;

            thread_init_data data(make_thread_function_nullary(pika::util::detail::bind(
                                      &lock_and_wait, std::ref(m[index]), std::ref(b0),
                                      std::ref(b1), std::ref(pikathreads[j]), wait)),
                "lock_and_wait");
            ids.push_back(register_thread(data));
        }

        // Tell all pikathreads that they can start running.
        b0.arrive_and_wait();

        // Wait for all pikathreads to finish.
        b1.arrive_and_wait();

        // {{{ Print results for this iteration.
        for (value_type& entry : pikathreads)
        {
            std::cout << "  " << entry.first << "," << entry.second << "\n";
        }
        // }}}
    }

    // Initiate shutdown of the runtime system.
    pika::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description desc_commandline("Usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
        ("pikathreads,T", value<std::size_t>()->default_value(128),
            "the number of PX threads to invoke")
        ("mutexes,M", value<std::size_t>()->default_value(1),
            "the number of mutexes to use")
        ("wait", value<std::size_t>()->default_value(30),
            "the number of milliseconds to wait between each lock attempt")
        ("iterations", value<std::size_t>()->default_value(1),
            "the number of times to repeat the test")
        ;
    // clang-format on

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return pika::init(pika_main, argc, argv, init_args);
}
