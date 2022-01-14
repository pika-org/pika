//  Copyright 2015 (c) Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case tests the workaround for the issue described in #1485:
// `ignore_while_checking` doesn't support all Lockable types.
// `ignore_all_while_checking` can be used instead to ignore all locks
// (including the ones that are not supported by `ignore_while_checking`).

#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/synchronization.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <functional>
#include <mutex>

struct wait_for_flag
{
    pika::lcos::local::spinlock mutex;
    pika::lcos::local::condition_variable_any cond_var;

    wait_for_flag()
      : flag(false)
      , woken(0)
    {
    }

    void wait(pika::lcos::local::spinlock& local_mutex,
        pika::lcos::local::condition_variable_any& local_cond_var, bool& running)
    {
        bool first = true;
        while (!flag)
        {
            // signal successful initialization
            if (first)
            {
                {
                    std::lock_guard<pika::lcos::local::spinlock> lk(local_mutex);
                    running = true;
                }

                first = false;
                local_cond_var.notify_all();
            }

            std::unique_lock<pika::lcos::local::spinlock> lk(mutex);
            if (!flag)
            {
                cond_var.wait(mutex);
            }
        }
        ++woken;
    }

    std::atomic<bool> flag;
    std::atomic<unsigned> woken;
};

void test_condition_with_mutex()
{
    wait_for_flag data;

    bool running = false;
    pika::lcos::local::spinlock local_mutex;
    pika::lcos::local::condition_variable_any local_cond_var;

    pika::thread thread(&wait_for_flag::wait, std::ref(data),
        std::ref(local_mutex), std::ref(local_cond_var), std::ref(running));

    // wait for the thread to run
    {
        std::unique_lock<pika::lcos::local::spinlock> lk(local_mutex);
        // NOLINTNEXTLINE(bugprone-infinite-loop)
        while (!running)
            local_cond_var.wait(lk);
    }

    // now start actual test
    data.flag.store(true);

    {
        std::lock_guard<pika::lcos::local::spinlock> lock(data.mutex);
        pika::util::ignore_all_while_checking il;
        PIKA_UNUSED(il);

        data.cond_var.notify_one();
    }

    thread.join();
    PIKA_TEST_EQ(data.woken, 1u);
}

int pika_main()
{
    test_condition_with_mutex();
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
