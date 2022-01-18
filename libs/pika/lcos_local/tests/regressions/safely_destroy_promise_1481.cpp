//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1481:
// Sync primitives safe destruction

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <chrono>
#include <thread>

void test_safe_destruction()
{
    pika::thread t;
    pika::future<void> outer;

    {
        pika::lcos::local::promise<void> p;
        pika::shared_future<void> inner = p.get_future().share();

        // Delay returning from p.set_value() below to destroy the promise
        // before set_value returns.
        outer = inner.then([](pika::shared_future<void>&&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        });

        // create a thread which will make the inner future ready
        t = pika::thread([&p]() { p.set_value(); });
        inner.get();
    }

    outer.get();
    t.join();
}

int pika_main()
{
    test_safe_destruction();
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
