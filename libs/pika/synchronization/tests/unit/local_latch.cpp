//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/latch.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

#define NUM_THREADS std::size_t(100)

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::size_t> num_threads(0);

///////////////////////////////////////////////////////////////////////////////
void test_count_down_and_wait(pika::lcos::local::latch& l)
{
    ++num_threads;

    PIKA_TEST(!l.is_ready());
    l.count_down_and_wait();
}

void test_count_down(pika::lcos::local::latch& l)
{
    ++num_threads;

    PIKA_TEST(!l.is_ready());
    l.count_down(NUM_THREADS);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    // count_down_and_wait
    {
        pika::lcos::local::latch l(NUM_THREADS + 1);
        PIKA_TEST(!l.is_ready());

        std::vector<pika::future<void>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
            results.push_back(
                pika::async(&test_count_down_and_wait, std::ref(l)));

        PIKA_TEST(!l.is_ready());

        // Wait for all threads to reach this point.
        l.count_down_and_wait();

        pika::wait_all(results);

        PIKA_TEST(l.is_ready());
        PIKA_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    // count_down
    {
        num_threads.store(0);

        pika::lcos::local::latch l(NUM_THREADS + 1);
        PIKA_TEST(!l.is_ready());

        pika::future<void> f = pika::async(&test_count_down, std::ref(l));

        PIKA_TEST(!l.is_ready());
        l.count_down_and_wait();

        f.get();

        PIKA_TEST(l.is_ready());
        PIKA_TEST_EQ(num_threads.load(), std::size_t(1));
    }

    // wait
    {
        num_threads.store(0);

        pika::lcos::local::latch l(NUM_THREADS);
        PIKA_TEST(!l.is_ready());

        std::vector<pika::future<void>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
            results.push_back(
                pika::async(&test_count_down_and_wait, std::ref(l)));

        pika::wait_all(results);

        l.wait();

        PIKA_TEST(l.is_ready());
        PIKA_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    // count_up
    {
        num_threads.store(0);

        pika::lcos::local::latch l(1);
        PIKA_TEST(!l.is_ready());

        std::vector<pika::future<void>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
        {
            l.count_up(1);
            results.push_back(
                pika::async(&test_count_down_and_wait, std::ref(l)));
        }

        PIKA_TEST(!l.is_ready());

        // Wait for all threads to reach this point.
        l.count_down_and_wait();

        pika::wait_all(results);

        PIKA_TEST(l.is_ready());
        PIKA_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    // reset
    {
        num_threads.store(0);

        pika::lcos::local::latch l(0);
        l.reset(NUM_THREADS + 1);
        PIKA_TEST(!l.is_ready());

        std::vector<pika::future<void>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
        {
            results.push_back(
                pika::async(&test_count_down_and_wait, std::ref(l)));
        }

        PIKA_TEST(!l.is_ready());

        // Wait for all threads to reach this point.
        l.count_down_and_wait();

        pika::wait_all(results);

        PIKA_TEST(l.is_ready());
        PIKA_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    PIKA_TEST_EQ(pika::local::finalize(), 0);
    return 0;
}

int main(int argc, char* argv[])
{
    // Initialize and run pika
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
