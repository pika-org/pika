//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/execution_base.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

std::size_t dummy_called = 0;

struct dummy_context : pika::execution::detail::context_base
{
    pika::execution::detail::resource_base const& resource() const override { return resource_; }

    pika::execution::detail::resource_base resource_;
};

struct dummy_agent : pika::execution::detail::agent_base
{
    std::string description() const override { return ""; }
    dummy_context const& context() const override { return context_; }

    void yield(char const*) override { ++dummy_called; }
    void yield_k(std::size_t, char const*) override {}
    void spin_k(std::size_t, char const*) override {}
    void suspend(char const*) override {}
    void resume(char const*) override {}
    void abort(char const*) override {}
    void sleep_for(pika::chrono::steady_duration const&, char const*) override {}
    void sleep_until(pika::chrono::steady_time_point const&, char const*) override {}

    dummy_context context_;
};

void test_basic_functionality()
{
    // Test that execution context forwards properly and resetting works
    {
        PIKA_TEST_EQ(dummy_called, 0u);
        {
            dummy_agent dummy;
            pika::execution::this_thread::detail::reset_agent ctx(dummy);
            pika::execution::this_thread::detail::yield();
        }

        PIKA_TEST_EQ(dummy_called, 1u);

        pika::execution::this_thread::detail::yield();

        PIKA_TEST_EQ(dummy_called, 1u);
    }

    // Test that we get different contexts in different threads...
    {
        auto context = pika::execution::this_thread::detail::agent();
        std::thread t([&context]() {
            PIKA_TEST_NEQ(context, pika::execution::this_thread::detail::agent());
        });
        t.join();
    }
}

struct simple_spinlock
{
    simple_spinlock() = default;

    void lock()
    {
        while (locked_.test_and_set()) { pika::execution::this_thread::detail::yield(); }
    }

    void unlock() { locked_.clear(); }

    std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
};

void test_yield()
{
    std::vector<std::thread> ts;
    simple_spinlock mutex;
    std::size_t counter = 0;
    std::size_t repetitions = 1000;
    for (std::size_t i = 0; i != static_cast<std::size_t>(std::thread::hardware_concurrency()) * 10;
         ++i)
    {
        ts.emplace_back([&mutex, &counter, repetitions]() {
            for (std::size_t repeat = 0; repeat != repetitions; ++repeat)
            {
                std::unique_lock<simple_spinlock> l(mutex);
                ++counter;
            }
        });
    }

    for (auto& t : ts) t.join();

    PIKA_TEST_EQ(counter, std::thread::hardware_concurrency() * repetitions * 10);
}

void test_suspend_resume()
{
    std::mutex mtx;
    pika::execution::detail::agent_ref suspended;

    bool resumed = false;

    std::thread t1([&mtx, &suspended, &resumed]() {
        auto context = pika::execution::this_thread::detail::agent();
        {
            std::unique_lock<std::mutex> l(mtx);
            suspended = context;
        }
        context.suspend();
        resumed = true;
    });

    while (true)
    {
        std::unique_lock<std::mutex> l(mtx);
        if (suspended) break;
    }

    suspended.resume();

    t1.join();
    PIKA_TEST(resumed);
}

void test_sleep()
{
    auto now = std::chrono::steady_clock::now();
    auto sleep_duration = std::chrono::milliseconds(100);
    pika::execution::this_thread::detail::sleep_for(sleep_duration);
    PIKA_TEST(now + sleep_duration <= std::chrono::steady_clock::now());

    auto sleep_time = sleep_duration * 2 + std::chrono::steady_clock::now();
    pika::execution::this_thread::detail::sleep_until(sleep_time);
    PIKA_TEST(now + sleep_duration * 2 <= std::chrono::steady_clock::now());
}

int main()
{
    test_basic_functionality();
    test_yield();
    test_suspend_resume();
    test_sleep();

    return 0;
}
