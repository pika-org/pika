//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
std::int32_t increment(std::int32_t i)
{
    return i + 1;
}

std::int32_t increment_with_future(pika::shared_future<std::int32_t> fi)
{
    return fi.get() + 1;
}

///////////////////////////////////////////////////////////////////////////////
struct mult2
{
    std::int32_t operator()(std::int32_t i) const
    {
        return i * 2;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct decrement
{
    std::int32_t call(std::int32_t i) const
    {
        return i - 1;
    }
};

///////////////////////////////////////////////////////////////////////////////
void do_nothing(std::int32_t) {}

struct do_nothing_obj
{
    void operator()(std::int32_t) const {}
};

struct do_nothing_member
{
    void call(std::int32_t) const {}
};

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_async_with_executor(Executor& exec)
{
    {
        pika::future<std::int32_t> f1 = pika::async(exec, &increment, 42);
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<void> f2 = pika::async(exec, &do_nothing, 42);
        f2.get();
    }

    {
        pika::lcos::local::promise<std::int32_t> p;
        pika::shared_future<std::int32_t> f = p.get_future();

        pika::future<std::int32_t> f1 =
            pika::async(exec, &increment_with_future, f);
        pika::future<std::int32_t> f2 =
            pika::async(exec, &increment_with_future, f);

        p.set_value(42);
        PIKA_TEST_EQ(f1.get(), 43);
        PIKA_TEST_EQ(f2.get(), 43);
    }

    {
        using pika::util::placeholders::_1;

        pika::future<std::int32_t> f1 =
            pika::async(exec, pika::util::bind(&increment, 42));
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<std::int32_t> f2 =
            pika::async(exec, pika::util::bind(&increment, _1), 42);
        PIKA_TEST_EQ(f2.get(), 43);
    }

    {
        pika::future<std::int32_t> f1 = pika::async(exec, increment, 42);
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<void> f2 = pika::async(exec, do_nothing, 42);
        f2.get();
    }

    {
        mult2 mult;

        pika::future<std::int32_t> f1 = pika::async(exec, mult, 42);
        PIKA_TEST_EQ(f1.get(), 84);
    }

    {
        mult2 mult;

        pika::future<std::int32_t> f1 =
            pika::async(exec, pika::util::bind(mult, 42));
        PIKA_TEST_EQ(f1.get(), 84);

        using pika::util::placeholders::_1;

        pika::future<std::int32_t> f2 =
            pika::async(exec, pika::util::bind(mult, _1), 42);
        PIKA_TEST_EQ(f2.get(), 84);

        do_nothing_obj do_nothing_f;
        pika::future<void> f3 =
            pika::async(exec, pika::util::bind(do_nothing_f, _1), 42);
        f3.get();
    }

    {
        decrement dec;

        pika::future<std::int32_t> f1 =
            pika::async(exec, &decrement::call, dec, 42);
        PIKA_TEST_EQ(f1.get(), 41);

        do_nothing_member dnm;
        pika::future<void> f2 =
            pika::async(exec, &do_nothing_member::call, dnm, 42);
        f2.get();
    }

    {
        decrement dec;

        using pika::util::placeholders::_1;

        pika::future<std::int32_t> f1 =
            pika::async(exec, pika::util::bind(&decrement::call, dec, 42));
        PIKA_TEST_EQ(f1.get(), 41);

        pika::future<std::int32_t> f2 =
            pika::async(exec, pika::util::bind(&decrement::call, dec, _1), 42);
        PIKA_TEST_EQ(f2.get(), 41);

        do_nothing_member dnm;
        pika::future<void> f3 = pika::async(
            exec, pika::util::bind(&do_nothing_member::call, dnm, _1), 42);
        f3.get();
    }
}

int pika_main()
{
    {
        pika::execution::sequenced_executor exec;
        test_async_with_executor(exec);
    }

    {
        pika::execution::parallel_executor exec;
        test_async_with_executor(exec);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run pika
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
