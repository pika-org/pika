//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/synchronization/channel_spsc.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

constexpr int NUM_WORKERS = 1000;

///////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T channel_get(pika::experimental::channel_spsc<T> const& c)
{
    T result;
    while (!c.get(&result)) { pika::this_thread::yield(); }
    return result;
}

template <typename T>
inline void channel_set(pika::experimental::channel_spsc<T>& c, T val)
{
    while (!c.set(std::move(val)))    // NOLINT
    {
        pika::this_thread::yield();
    }
}

///////////////////////////////////////////////////////////////////////////////
int thread_func(int i, pika::experimental::channel_spsc<int>& channel,
    pika::experimental::channel_spsc<int>& next)
{
    channel_set(channel, i);
    return channel_get(next);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::vector<pika::experimental::channel_spsc<int>> channels;
    channels.reserve(NUM_WORKERS);

    std::vector<ex::unique_any_sender<int>> workers;
    workers.reserve(NUM_WORKERS);

    for (int i = 0; i != NUM_WORKERS; ++i) { channels.emplace_back(std::size_t(1)); }

    auto sched = ex::thread_pool_scheduler{};
    for (int i = 0; i != NUM_WORKERS; ++i)
    {
        workers.emplace_back(ex::transfer_just(sched, i, std::ref(channels[i]),
                                 std::ref(channels[(i + 1) % NUM_WORKERS])) |
            ex::then(thread_func));
    }

    auto results = tt::sync_wait(ex::when_all_vector(std::move(workers)));

    for (int i = 0; i != NUM_WORKERS; ++i) { PIKA_TEST_EQ((i + 1) % NUM_WORKERS, results[i]); }

    pika::finalize();
    return 0;
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
