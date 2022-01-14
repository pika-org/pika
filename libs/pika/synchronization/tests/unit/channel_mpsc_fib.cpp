//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>
#include <pika/synchronization/channel_mpsc.hpp>

#include <functional>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int verify_fibonacci(int n)
{
    if (n < 2)
        return n;
    return verify_fibonacci(n - 1) + verify_fibonacci(n - 2);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T channel_get(pika::lcos::local::channel_mpsc<T> const& c)
{
    T result;
    while (!c.get(&result))
    {
        pika::this_thread::yield();
    }
    return result;
}

template <typename T>
inline void channel_set(pika::lcos::local::channel_mpsc<T>& c, T val)
{
    while (!c.set(std::move(val)))    // NOLINT
    {
        pika::this_thread::yield();
    }
}

///////////////////////////////////////////////////////////////////////////////
void produce_numbers(pika::lcos::local::channel_mpsc<int>& c2,
    pika::lcos::local::channel_mpsc<int>& c3)
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

void consume_numbers(int n, pika::lcos::local::channel_mpsc<bool>& c1,
    pika::lcos::local::channel_mpsc<int>& c2,
    pika::lcos::local::channel_mpsc<int>& c3)
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
    pika::lcos::local::channel_mpsc<bool> c1(1);
    pika::lcos::local::channel_mpsc<int> c2(1);
    pika::lcos::local::channel_mpsc<int> c3(5);

    pika::future<void> producer =
        pika::async(&produce_numbers, std::ref(c2), std::ref(c3));

    pika::future<void> consumer = pika::async(
        &consume_numbers, 22, std::ref(c1), std::ref(c2), std::ref(c3));

    pika::wait_all(producer, consumer);

    PIKA_TEST(channel_get(c1));

    pika::local::finalize();
    return pika::util::report_errors();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
