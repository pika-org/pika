//  Copyright 2013 (c) Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #878: `future::unwrap`
// triggers assertion

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <exception>
#include <utility>

int pika_main()
{
    pika::lcos::local::promise<pika::future<int>> promise;
    pika::future<pika::future<int>> future = promise.get_future();
    std::exception_ptr p;
    try
    {
        //promise.set_value(42);
        throw pika::bad_parameter;
    }
    catch (...)
    {
        p = std::current_exception();
    }
    PIKA_TEST(p);
    promise.set_exception(std::move(p));
    PIKA_TEST(future.has_exception());

    pika::future<int> inner(std::move(future));
    PIKA_TEST(inner.has_exception());

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
