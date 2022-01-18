//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    pika::shared_future<int> f1 = pika::make_ready_future(42);

    pika::future<int> f2 = f1.then(
        [](pika::shared_future<int>&&) { return pika::make_ready_future(43); });

    PIKA_TEST_EQ(f1.get(), 42);
    PIKA_TEST_EQ(f2.get(), 43);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run pika
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
