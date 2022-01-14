//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
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
int pika_main()
{
    {
        pika::future<std::int32_t> f1 = pika::async(&increment, 42);
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<std::int32_t> f2 =
            pika::async(pika::launch::all, &increment, 42);
        PIKA_TEST_EQ(f2.get(), 43);

        pika::future<void> f3 = pika::async(&do_nothing, 42);
        f3.get();

        pika::future<void> f4 = pika::async(pika::launch::sync, &do_nothing, 42);
        f4.get();
    }

    {
        pika::lcos::local::promise<std::int32_t> p;
        pika::shared_future<std::int32_t> f = p.get_future();

        pika::future<std::int32_t> f1 = pika::async(&increment_with_future, f);
        pika::future<std::int32_t> f2 =
            pika::async(pika::launch::all, &increment_with_future, f);

        p.set_value(42);
        PIKA_TEST_EQ(f1.get(), 43);
        PIKA_TEST_EQ(f2.get(), 43);
    }

    {
        using pika::util::placeholders::_1;

        pika::future<std::int32_t> f1 =
            pika::async(pika::util::bind(&increment, 42));
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<std::int32_t> f2 =
            pika::async(pika::launch::all, pika::util::bind(&increment, _1), 42);
        PIKA_TEST_EQ(f2.get(), 43);

        pika::future<std::int32_t> f3 =
            pika::async(pika::util::bind(&increment, 42));
        PIKA_TEST_EQ(f3.get(), 43);

        pika::future<std::int32_t> f4 =
            pika::async(pika::launch::all, pika::util::bind(&increment, _1), 42);
        PIKA_TEST_EQ(f4.get(), 43);

        pika::future<void> f5 =
            pika::async(pika::launch::all, pika::util::bind(&do_nothing, _1), 42);
        f5.get();

        pika::future<void> f6 =
            pika::async(pika::launch::sync, pika::util::bind(&do_nothing, _1), 42);
        f6.get();
    }

    {
        pika::future<std::int32_t> f1 = pika::async(increment, 42);
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<std::int32_t> f2 =
            pika::async(pika::launch::all, increment, 42);
        PIKA_TEST_EQ(f2.get(), 43);

        pika::future<void> f3 = pika::async(do_nothing, 42);
        f3.get();

        pika::future<void> f4 = pika::async(pika::launch::sync, do_nothing, 42);
        f4.get();
    }

    {
        using pika::util::placeholders::_1;

        pika::future<std::int32_t> f1 =
            pika::async(pika::util::bind(increment, 42));
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<std::int32_t> f2 =
            pika::async(pika::launch::all, pika::util::bind(increment, _1), 42);
        PIKA_TEST_EQ(f2.get(), 43);

        pika::future<std::int32_t> f3 =
            pika::async(pika::util::bind(increment, 42));
        PIKA_TEST_EQ(f3.get(), 43);

        pika::future<std::int32_t> f4 =
            pika::async(pika::launch::all, pika::util::bind(increment, _1), 42);
        PIKA_TEST_EQ(f4.get(), 43);

        pika::future<void> f5 =
            pika::async(pika::launch::all, pika::util::bind(do_nothing, _1), 42);
        f5.get();

        pika::future<void> f6 =
            pika::async(pika::launch::sync, pika::util::bind(do_nothing, _1), 42);
        f6.get();
    }

    {
        mult2 mult;

        pika::future<std::int32_t> f1 = pika::async(mult, 42);
        PIKA_TEST_EQ(f1.get(), 84);

        pika::future<std::int32_t> f2 = pika::async(pika::launch::all, mult, 42);
        PIKA_TEST_EQ(f2.get(), 84);
    }

    {
        mult2 mult;

        pika::future<std::int32_t> f1 = pika::async(pika::util::bind(mult, 42));
        PIKA_TEST_EQ(f1.get(), 84);

        using pika::util::placeholders::_1;

        pika::future<std::int32_t> f2 =
            pika::async(pika::launch::all, pika::util::bind(mult, 42));
        PIKA_TEST_EQ(f2.get(), 84);

        pika::future<std::int32_t> f3 =
            pika::async(pika::util::bind(mult, _1), 42);
        PIKA_TEST_EQ(f3.get(), 84);

        pika::future<std::int32_t> f4 =
            pika::async(pika::launch::all, pika::util::bind(mult, _1), 42);
        PIKA_TEST_EQ(f4.get(), 84);

        do_nothing_obj do_nothing_f;
        pika::future<void> f5 =
            pika::async(pika::launch::all, pika::util::bind(do_nothing_f, _1), 42);
        f5.get();

        pika::future<void> f6 = pika::async(
            pika::launch::sync, pika::util::bind(do_nothing_f, _1), 42);
        f6.get();
    }

    {
        decrement dec;

        pika::future<std::int32_t> f1 = pika::async(&decrement::call, dec, 42);
        PIKA_TEST_EQ(f1.get(), 41);

        pika::future<std::int32_t> f2 =
            pika::async(pika::launch::all, &decrement::call, dec, 42);
        PIKA_TEST_EQ(f2.get(), 41);

        do_nothing_member dnm;
        pika::future<void> f3 =
            pika::async(pika::launch::all, &do_nothing_member::call, dnm, 42);
        f3.get();

        pika::future<void> f4 =
            pika::async(pika::launch::sync, &do_nothing_member::call, dnm, 42);
        f4.get();
    }

    {
        decrement dec;

        using pika::util::placeholders::_1;

        pika::future<std::int32_t> f1 =
            pika::async(pika::util::bind(&decrement::call, dec, 42));
        PIKA_TEST_EQ(f1.get(), 41);

        pika::future<std::int32_t> f2 = pika::async(
            pika::launch::all, pika::util::bind(&decrement::call, dec, 42));
        PIKA_TEST_EQ(f2.get(), 41);

        pika::future<std::int32_t> f3 =
            pika::async(pika::util::bind(&decrement::call, dec, _1), 42);
        PIKA_TEST_EQ(f3.get(), 41);

        pika::future<std::int32_t> f4 = pika::async(
            pika::launch::all, pika::util::bind(&decrement::call, dec, _1), 42);
        PIKA_TEST_EQ(f4.get(), 41);

        do_nothing_member dnm;
        pika::future<void> f5 = pika::async(pika::launch::all,
            pika::util::bind(&do_nothing_member::call, dnm, _1), 42);
        f5.get();

        pika::future<void> f6 = pika::async(pika::launch::sync,
            pika::util::bind(&do_nothing_member::call, dnm, _1), 42);
        f6.get();
    }

    {
        using pika::util::placeholders::_1;

        auto policy1 = pika::launch::select([]() { return pika::launch::sync; });

        pika::future<std::int32_t> f1 =
            pika::async(policy1, pika::util::bind(&increment, 42));
        PIKA_TEST_EQ(f1.get(), 43);

        pika::future<std::int32_t> f2 =
            pika::async(policy1, pika::util::bind(&increment, _1), 42);
        PIKA_TEST_EQ(f2.get(), 43);

        std::atomic<int> count(0);
        auto policy2 = pika::launch::select([&count]() -> pika::launch {
            if (count++ == 0)
                return pika::launch::async;
            return pika::launch::sync;
        });

        pika::future<void> f3 =
            pika::async(policy2, pika::util::bind(&do_nothing, _1), 42);
        f3.get();

        pika::future<void> f4 =
            pika::async(policy2, pika::util::bind(&do_nothing, 42));
        f4.get();
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
