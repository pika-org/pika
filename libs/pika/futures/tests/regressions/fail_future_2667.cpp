//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #2667: Ambiguity of
// nested pika::future<void>'s.
//
// This test is supposed to fail compiling.

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>

#include <chrono>
#include <utility>

int pika_main()
{
    pika::future<pika::future<int>> fut = pika::async([]() -> pika::future<int> {
        return pika::async([]() -> int { return 42; });
    });

    pika::future<void> fut2 = std::move(fut);
    fut2.get();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
