//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/synchronization/channel_mpsc.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <functional>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
int verify_fibonacci(int n)
{
    if (n < 2) return n;
    return verify_fibonacci(n - 1) + verify_fibonacci(n - 2);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T channel_get(pika::experimental::channel_mpsc<T> const& c)
{
    T result;
    while (!c.get(&result)) { pika::this_thread::yield(); }
    return result;
}

template <typename T>
inline void channel_set(pika::experimental::channel_mpsc<T>& c, T val)
{
    while (!c.set(std::move(val)))    // NOLINT
    {
        pika::this_thread::yield();
    }
}

///////////////////////////////////////////////////////////////////////////////
void produce_numbers(
    pika::experimental::channel_mpsc<int>& c2, pika::experimental::channel_mpsc<int>& c3)
{
    int f1 = 1, f2 = 0;

    int n = channel_get(c2);

    for (int i = 0; i <= n; ++i)
    {
        if (i < 2)
        {
            channel_set(c3, i);
            continue;
        }

        int fib = f1 + f2;
        f2 = f1;
        f1 = fib;

        channel_set(c3, fib);
    }
}

void consume_numbers(int n, pika::experimental::channel_mpsc<bool>& c1,
    pika::experimental::channel_mpsc<int>& c2, pika::experimental::channel_mpsc<int>& c3)
{
    channel_set(c2, n);

    for (int i = 0; i <= n; ++i)
    {
        int fib = channel_get(c3);
        PIKA_TEST_EQ(fib, verify_fibonacci(i));
    }

    channel_set(c1, true);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    pika::experimental::channel_mpsc<bool> c1(1);
    pika::experimental::channel_mpsc<int> c2(1);
    pika::experimental::channel_mpsc<int> c3(5);

    auto sched = ex::thread_pool_scheduler{};
    tt::sync_wait(
        ex::when_all(ex::schedule(sched) | ex::then([&]() mutable { produce_numbers(c2, c3); }),
            ex::schedule(sched) | ex::then([&]() mutable { consume_numbers(22, c1, c2, c3); })));

    PIKA_TEST(channel_get(c1));

    pika::finalize();
    return 0;
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
