//  Copyright (c) 2015-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1613: Dataflow causes
// stack overflow

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

#define NUM_FUTURES std::size_t(2 * PIKA_CONTINUATION_MAX_RECURSION_DEPTH)

void test_exception_from_continuation1()
{
    pika::lcos::local::promise<void> p;
    pika::future<void> f1 = p.get_future();

    pika::future<void> f2 = f1.then([](pika::future<void>&& f1) {
        PIKA_TEST(f1.has_value());
        PIKA_THROW_EXCEPTION(
            pika::invalid_status, "lambda", "testing exceptions");
    });

    p.set_value();
    f2.wait();

    PIKA_TEST(f2.has_exception());
}

void test_exception_from_continuation2()
{
    pika::lcos::local::promise<void> p;

    std::vector<pika::shared_future<void>> results;
    results.reserve(NUM_FUTURES + 1);

    std::atomic<std::size_t> recursion_level(0);
    std::atomic<std::size_t> exceptions_thrown(0);

    results.push_back(p.get_future());
    for (std::size_t i = 0; i != NUM_FUTURES; ++i)
    {
        results.push_back(
            results.back().then([&](pika::shared_future<void>&& f) {
                ++recursion_level;

                f.get();    // rethrow, if has exception

                ++exceptions_thrown;
                PIKA_THROW_EXCEPTION(
                    pika::invalid_status, "lambda", "testing exceptions");
            }));
    }

    // make futures ready in backwards sequence
    pika::apply([&p]() { p.set_value(); });
    pika::wait_all_nothrow(results);

    PIKA_TEST_EQ(recursion_level.load(), NUM_FUTURES);
    PIKA_TEST_EQ(exceptions_thrown.load(), std::size_t(1));

    // first future is the only one which does not hold exception
    PIKA_TEST(!results[0].has_exception());

    for (std::size_t i = 1; i != results.size(); ++i)
    {
        PIKA_TEST(results[i].has_exception());
    }
}

int pika_main()
{
    test_exception_from_continuation1();
    test_exception_from_continuation2();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init(pika_main, argc, argv);
    return pika::util::report_errors();
}
