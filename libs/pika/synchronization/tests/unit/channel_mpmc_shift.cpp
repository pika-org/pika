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
#include <pika/synchronization/channel_mpmc.hpp>

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

constexpr int NUM_WORKERS = 1000;

///////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T channel_get(pika::lcos::local::channel_mpmc<T> const& c)
{
    T result;
    while (!c.get(&result))
    {
        pika::this_thread::yield();
    }
    return result;
}

template <typename T>
inline void channel_set(pika::lcos::local::channel_mpmc<T>& c, T val)
{
    while (!c.set(std::move(val)))    // NOLINT
    {
        pika::this_thread::yield();
    }
}

///////////////////////////////////////////////////////////////////////////////
int thread_func(int i, pika::lcos::local::channel_mpmc<int>& channel,
    pika::lcos::local::channel_mpmc<int>& next)
{
    channel_set(channel, i);
    return channel_get(next);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::vector<pika::lcos::local::channel_mpmc<int>> channels;
    channels.reserve(NUM_WORKERS);

    std::vector<pika::future<int>> workers;
    workers.reserve(NUM_WORKERS);

    for (int i = 0; i != NUM_WORKERS; ++i)
    {
        channels.emplace_back(std::size_t(1));
    }

    for (int i = 0; i != NUM_WORKERS; ++i)
    {
        workers.push_back(pika::async(&thread_func, i, std::ref(channels[i]),
            std::ref(channels[(i + 1) % NUM_WORKERS])));
    }

    pika::wait_all(workers);

    for (int i = 0; i != NUM_WORKERS; ++i)
    {
        PIKA_TEST_EQ((i + 1) % NUM_WORKERS, workers[i].get());
    }

    pika::local::finalize();
    return pika::util::report_errors();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
