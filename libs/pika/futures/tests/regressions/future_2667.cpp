//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #2667: Ambiguity of
// nested pika::future<void>'s

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <utility>

std::atomic<bool> was_run(false);

void do_more_work()
{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    was_run = true;
}

int pika_main()
{
    pika::future<pika::future<void>> fut = pika::async([]() -> pika::future<void> {
        return pika::async([]() -> void { do_more_work(); });
    });

    pika::chrono::high_resolution_timer t;

    pika::future<void> fut2 = std::move(fut);
    fut2.get();

    PIKA_TEST_LT(1.0, t.elapsed());
    PIKA_TEST(was_run.load());

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
