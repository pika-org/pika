//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/timing.hpp>
#include <pika/synchronization/channel_spsc.hpp>
#include <pika/thread.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
struct data
{
    data() = default;

    explicit data(int d) { data_[0] = d; }

    int data_[8];
};

#if defined(PIKA_HAVE_VERIFY_LOCKS)
constexpr int NUM_TESTS = 10000;
#elif PIKA_DEBUG
constexpr int NUM_TESTS = 1000000;
#else
constexpr int NUM_TESTS = 100000000;
#endif

///////////////////////////////////////////////////////////////////////////////
inline data channel_get(pika::experimental::channel_spsc<data> const& c)
{
    data result;
    while (!c.get(&result)) { pika::this_thread::yield(); }
    return result;
}

inline void channel_set(pika::experimental::channel_spsc<data>& c, data&& val)
{
    while (!c.set(std::move(val)))    // NOLINT
    {
        pika::this_thread::yield();
    }
}

///////////////////////////////////////////////////////////////////////////////
// Produce
double thread_func_0(pika::experimental::channel_spsc<data>& c)
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i != NUM_TESTS; ++i) { channel_set(c, data{i}); }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end - start).count();
}

// Consume
double thread_func_1(pika::experimental::channel_spsc<data>& c)
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i != NUM_TESTS; ++i)
    {
        data d = channel_get(c);
        if (d.data_[0] != i) { std::cout << "Error!\n"; }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end - start).count();
}

int pika_main()
{
    auto sched = ex::thread_pool_scheduler{};

    pika::experimental::channel_spsc<data> c(10000);

    ex::unique_any_sender<double> producer =
        ex::transfer_just(sched, std::ref(c)) | ex::then(thread_func_0) | ex::ensure_started();
    ex::unique_any_sender<double> consumer =
        ex::transfer_just(sched, std::ref(c)) | ex::then(thread_func_1) | ex::ensure_started();

    auto producer_time = tt::sync_wait(std::move(producer));
    std::cout << "Producer throughput: " << (NUM_TESTS / producer_time) << " [op/s] ("
              << (producer_time / NUM_TESTS) << " [s/op])\n";

    auto consumer_time = tt::sync_wait(std::move(consumer));
    std::cout << "Consumer throughput: " << (NUM_TESTS / consumer_time) << " [op/s] ("
              << (consumer_time / NUM_TESTS) << " [s/op])\n";

    return pika::finalize();
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
