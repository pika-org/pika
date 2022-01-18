//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Making sure the continuations of a shared_future are invoked in the same
// order as they have been attached.

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>

std::atomic<int> invocation_count(0);

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    pika::lcos::local::promise<int> p;
    pika::shared_future<int> f1 = p.get_future();

    pika::future<int> f2 = f1.then([](pika::shared_future<int>&& f) {
        PIKA_TEST_EQ(f.get(), 42);
        return ++invocation_count;
    });

    pika::future<int> f3 = f1.then([](pika::shared_future<int>&& f) {
        PIKA_TEST_EQ(f.get(), 42);
        return ++invocation_count;
    });

    p.set_value(42);

    PIKA_TEST_EQ(f1.get(), 42);
    PIKA_TEST_EQ(f2.get(), 1);
    PIKA_TEST_EQ(f3.get(), 2);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
