//  Copyright (c) 2011-2013 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue described in #1036: Scheduler hangs when
// user code attempts to "block" OS-threads

#include <pika/functional/bind.hpp>
#include <pika/local/barrier.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/timing.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/topology/topology.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void blocker(pika::barrier<>& exit_barrier, std::atomic<std::uint64_t>& entered,
    std::atomic<std::uint64_t>& started,
    std::unique_ptr<std::atomic<std::uint64_t>[]>& blocked_threads,
    std::uint64_t worker)
{
    // reschedule if we are not on the correct OS thread...
    if (worker != pika::get_worker_thread_num())
    {
        pika::threads::thread_init_data data(
            pika::threads::make_thread_function_nullary(pika::util::bind(&blocker,
                std::ref(exit_barrier), std::ref(entered), std::ref(started),
                std::ref(blocked_threads), worker)),
            "blocker", pika::threads::thread_priority::normal,
            pika::threads::thread_schedule_hint(worker));
        pika::threads::register_work(data);
        return;
    }

    blocked_threads[pika::get_worker_thread_num()].fetch_add(1);

    entered.fetch_add(1);

    PIKA_TEST_EQ(worker, pika::get_worker_thread_num());

    while (started.load() != 1)
        continue;

    exit_barrier.arrive_and_drop();
}

///////////////////////////////////////////////////////////////////////////////
std::uint64_t delay = 100;

int pika_main()
{
    {
        ///////////////////////////////////////////////////////////////////////
        // Block all other OS threads.
        std::atomic<std::uint64_t> entered(0);
        std::atomic<std::uint64_t> started(0);

        std::uint64_t const os_thread_count = pika::get_os_thread_count();

        pika::barrier<> exit_barrier(os_thread_count);

        std::unique_ptr<std::atomic<std::uint64_t>[]> blocked_threads(
            new std::atomic<std::uint64_t>[os_thread_count]);

        for (std::uint64_t i = 0; i < os_thread_count; ++i)
            blocked_threads[i].store(0);

        std::uint64_t scheduled = 0;
        for (std::uint64_t i = 0; i < os_thread_count; ++i)
        {
            if (i == pika::get_worker_thread_num())
                continue;

            pika::threads::thread_init_data data(
                pika::threads::make_thread_function_nullary(pika::util::bind(
                    &blocker, std::ref(exit_barrier), std::ref(entered),
                    std::ref(started), std::ref(blocked_threads), i)),
                "blocker", pika::threads::thread_priority::normal,
                pika::threads::thread_schedule_hint(i));
            pika::threads::register_work(data);
            ++scheduled;
        }
        PIKA_TEST_EQ(scheduled, os_thread_count - 1);

        while (entered.load() != (os_thread_count - 1))
            continue;

        {
            double delay_sec = delay * 1e-6;
            pika::chrono::high_resolution_timer td;

            while (true)
            {
                if (td.elapsed() > delay_sec)
                    break;
            }
        }

        started.fetch_add(1);

        for (std::uint64_t i = 0; i < os_thread_count; ++i)
            PIKA_TEST_LTE(blocked_threads[i].load(), std::uint64_t(1));

        exit_barrier.arrive_and_wait();
    }

    return pika::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using namespace pika::program_options;

    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    cmdline.add_options()("delay",
        value<std::uint64_t>(&delay)->default_value(100),
        "time in micro-seconds for the delay loop");

    // We force this test to use all available threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika.
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
