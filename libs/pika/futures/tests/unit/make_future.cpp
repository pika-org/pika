//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/execution.hpp>
#include <pika/modules/futures.hpp>
#include <pika/modules/testing.hpp>

#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
void test_make_future()
{
    // test make_future<T>(future<T>)
    {
        pika::future<int> f1 = pika::make_ready_future(42);
        pika::future<int> f2 = pika::make_future<int>(std::move(f1));
        PIKA_TEST_EQ(42, f2.get());
    }

    // test make_future<T>(future<U>) where is_convertible<U, T>
    {
        pika::future<int> f1 = pika::make_ready_future(42);
        pika::future<double> f2 = pika::make_future<double>(std::move(f1));
        PIKA_TEST_EQ(42.0, f2.get());
    }

    // test make_future<void>(future<U>)
    {
        pika::future<int> f1 = pika::make_ready_future(42);
        pika::future<void> f2 = pika::make_future<void>(std::move(f1));
    }

    // test make_future<void>(future<void>)
    {
        pika::future<void> f1 = pika::make_ready_future();
        pika::future<void> f2 = pika::make_future<void>(std::move(f1));
    }

    // test make_future<T>(future<U>) with given T conv(U)
    {
        pika::future<int> f1 = pika::make_ready_future(42);
        pika::future<std::string> f2 =
            pika::make_future<std::string>(std::move(f1),
                [](int value) -> std::string { return std::to_string(value); });

        PIKA_TEST_EQ(std::string("42"), f2.get());
    }
}

void test_make_shared_future()
{
    // test make_future<T>(shared_future<T>)
    {
        pika::shared_future<int> f1 = pika::make_ready_future(42);
        pika::shared_future<int> f2 = pika::make_future<int>(f1);
        PIKA_TEST_EQ(42, f1.get());
        PIKA_TEST_EQ(42, f2.get());
    }

    // test make_future<T>(shared_future<U>) where is_convertible<U, T>
    {
        pika::shared_future<int> f1 = pika::make_ready_future(42);
        pika::shared_future<double> f2 = pika::make_future<double>(f1);
        PIKA_TEST_EQ(42, f1.get());
        PIKA_TEST_EQ(42.0, f2.get());
    }

    // test make_future<void>(shared_future<U>)
    {
        pika::shared_future<int> f1 = pika::make_ready_future(42);
        pika::shared_future<void> f2 = pika::make_future<void>(f1);
        PIKA_TEST_EQ(42, f1.get());
    }

    // test make_future<void>(shared_future<void>)
    {
        pika::shared_future<void> f1 = pika::make_ready_future();
        pika::shared_future<void> f2 = pika::make_future<void>(f1);
    }

    // test make_future<T>(shared_future<U>) with given T conv(U)
    {
        pika::shared_future<int> f1 = pika::make_ready_future(42);
        pika::shared_future<std::string> f2 = pika::make_future<std::string>(
            f1, [](int value) -> std::string { return std::to_string(value); });

        PIKA_TEST_EQ(42, f1.get());
        PIKA_TEST_EQ(std::string("42"), f2.get());
    }
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_make_future();
    test_make_shared_future();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv), 0);
    return pika::util::report_errors();
}
