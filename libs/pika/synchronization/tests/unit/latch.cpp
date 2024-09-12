//  Copyright (c) 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/latch.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

#define NUM_THREADS std::size_t(100)

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::size_t> num_threads(0);

///////////////////////////////////////////////////////////////////////////////
void test_arrive_and_wait(pika::latch& l)
{
    ++num_threads;

    PIKA_TEST(!l.try_wait());
    l.arrive_and_wait();
}

void test_count_down(pika::latch& l)
{
    ++num_threads;

    PIKA_TEST(!l.try_wait());
    l.count_down(NUM_THREADS);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    auto sched = ex::thread_pool_scheduler{};

    // arrive_and_wait
    {
        pika::latch l(NUM_THREADS + 1);
        PIKA_TEST(!l.try_wait());

        std::vector<ex::unique_any_sender<>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
        {
            results.emplace_back(ex::just(std::ref(l)) | ex::continues_on(sched) |
                ex::then(test_arrive_and_wait) | ex::ensure_started());
        }

        PIKA_TEST(!l.try_wait());

        // Wait for all threads to reach this point.
        l.arrive_and_wait();

        tt::sync_wait(ex::when_all_vector(std::move(results)));

        PIKA_TEST(l.try_wait());
        PIKA_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    // count_down
    {
        num_threads.store(0);

        pika::latch l(NUM_THREADS + 1);
        PIKA_TEST(!l.try_wait());

        auto s = ex::just(std::ref(l)) | ex::continues_on(sched) | ex::then(test_count_down) |
            ex::ensure_started();

        PIKA_TEST(!l.try_wait());
        l.arrive_and_wait();

        tt::sync_wait(std::move(s));

        PIKA_TEST(l.try_wait());
        PIKA_TEST_EQ(num_threads.load(), std::size_t(1));
    }

    // wait
    {
        num_threads.store(0);

        pika::latch l(NUM_THREADS);
        PIKA_TEST(!l.try_wait());

        std::vector<ex::unique_any_sender<>> results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
        {
            results.emplace_back(ex::just(std::ref(l)) | ex::continues_on(sched) |
                ex::then(test_arrive_and_wait) | ex::ensure_started());
        }

        tt::sync_wait(ex::when_all_vector(std::move(results)));

        l.wait();

        PIKA_TEST(l.try_wait());
        PIKA_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    // Initialize and run pika
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0, "pika main exited with non-zero status");

    return 0;
}
