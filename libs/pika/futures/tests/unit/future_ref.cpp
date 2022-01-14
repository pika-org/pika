// Copyright (C) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <chrono>
#include <functional>

int global;

int& foo()
{
    return global;
}

void test_make_ready_future()
{
    pika::future<int&> f = pika::make_ready_future(std::ref(global));
    PIKA_TEST_EQ(&f.get(), &global);

#if 0    // Timed make_ready_future is not supported
    pika::future<int&> f_at = pika::make_ready_future_at(
        std::chrono::system_clock::now() + std::chrono::seconds(1),
        std::ref(global));
    PIKA_TEST_EQ(&f_at.get(), &global);

    pika::future<int&> f_after =
        pika::make_ready_future_after(std::chrono::seconds(1), std::ref(global));
    PIKA_TEST_EQ(&f_after.get(), &global);
#endif
}

void test_async()
{
    pika::future<int&> f = pika::async(&foo);
    PIKA_TEST_EQ(&f.get(), &global);

    pika::future<int&> f_sync = pika::async(pika::launch::sync, &foo);
    PIKA_TEST_EQ(&f_sync.get(), &global);
}

int pika_main()
{
    test_make_ready_future();
    test_async();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
