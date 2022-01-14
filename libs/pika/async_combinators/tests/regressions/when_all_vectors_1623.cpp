//  Copyright 2015 (c) Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1623: pika::wait_all()
// invoked with two vector<future<T>> fails

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_when_all()
{
    std::vector<pika::future<void>> v1, v2;
    v1.push_back(pika::make_ready_future());
    v2.push_back(pika::make_ready_future());

    pika::when_all(v1, v2).get();
}

void test_when_any()
{
    std::vector<pika::future<void>> v1, v2;
    v1.push_back(pika::make_ready_future());
    v2.push_back(pika::make_ready_future());

    pika::when_any(v1, v2).get();
}

void test_when_some()
{
    std::vector<pika::future<void>> v1, v2;
    v1.push_back(pika::make_ready_future());
    v2.push_back(pika::make_ready_future());

    pika::when_some(1, v1, v2).get();
}

///////////////////////////////////////////////////////////////////////////////
void test_wait_all()
{
    std::vector<pika::future<void>> v1, v2;
    v1.push_back(pika::make_ready_future());
    v2.push_back(pika::make_ready_future());

    pika::wait_all(v1, v2);
}

void test_wait_any()
{
    std::vector<pika::future<void>> v1, v2;
    v1.push_back(pika::make_ready_future());
    v2.push_back(pika::make_ready_future());

    pika::wait_any(v1, v2);
}

void test_wait_some()
{
    std::vector<pika::future<void>> v1, v2;
    v1.push_back(pika::make_ready_future());
    v2.push_back(pika::make_ready_future());

    pika::wait_some(1, v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_when_all();
    test_when_any();
    test_when_some();

    test_wait_all();
    test_wait_any();
    test_wait_some();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
