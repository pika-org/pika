//  Copyright (C) 2012-2013 Vicente Botet
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int p1()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 1;
}

int p2(pika::future<int> f)
{
    PIKA_TEST(f.valid());
    int i = f.get();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 2 * i;
}

void p3(pika::future<int> f)
{
    PIKA_TEST(f.valid());
    int i = f.get();
    (void) i;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return;
}

pika::future<int> p4(pika::future<int> f)
{
    return pika::async(p2, std::move(f));
}

///////////////////////////////////////////////////////////////////////////////
void test_return_int()
{
    pika::future<int> f1 = pika::async(pika::launch::async, &p1);
    PIKA_TEST(f1.valid());
    pika::future<int> f2 = f1.then(&p2);
    PIKA_TEST(f2.valid());
    try
    {
        PIKA_TEST_EQ(f2.get(), 2);
    }
    catch (pika::exception const& /*ex*/)
    {
        PIKA_TEST(false);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

void test_return_int_launch()
{
    pika::future<int> f1 = pika::async(pika::launch::async, &p1);
    PIKA_TEST(f1.valid());
    pika::future<int> f2 = f1.then(pika::launch::async, &p2);
    PIKA_TEST(f2.valid());
    try
    {
        PIKA_TEST_EQ(f2.get(), 2);
    }
    catch (pika::exception const& /*ex*/)
    {
        PIKA_TEST(false);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_return_void()
{
    pika::future<int> f1 = pika::async(pika::launch::async, &p1);
    PIKA_TEST(f1.valid());
    pika::future<void> f2 = f1.then(&p3);
    PIKA_TEST(f2.valid());
    try
    {
        f2.wait();
    }
    catch (pika::exception const& /*ex*/)
    {
        PIKA_TEST(false);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

void test_return_void_launch()
{
    pika::future<int> f1 = pika::async(pika::launch::async, &p1);
    PIKA_TEST(f1.valid());
    pika::future<void> f2 = f1.then(pika::launch::sync, &p3);
    PIKA_TEST(f2.valid());
    try
    {
        f2.wait();
    }
    catch (pika::exception const& /*ex*/)
    {
        PIKA_TEST(false);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_implicit_unwrapping()
{
    pika::future<int> f1 = pika::async(pika::launch::async, &p1);
    PIKA_TEST(f1.valid());
    pika::future<int> f2 = f1.then(&p4);
    PIKA_TEST(f2.valid());
    try
    {
        PIKA_TEST(f2.get() == 2);
    }
    catch (pika::exception const& /*ex*/)
    {
        PIKA_TEST(false);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_simple_then()
{
    pika::future<int> f2 = pika::async(p1).then(&p2);
    PIKA_TEST(f2.get() == 2);
}

void test_simple_deferred_then()
{
    pika::future<int> f2 = pika::async(pika::launch::deferred, p1).then(&p2);
    PIKA_TEST(f2.get() == 2);
}

///////////////////////////////////////////////////////////////////////////////
void test_complex_then()
{
    pika::future<int> f1 = pika::async(p1);
    pika::future<int> f21 = f1.then(&p2);
    pika::future<int> f2 = f21.then(&p2);
    PIKA_TEST_EQ(f2.get(), 4);
}

void test_complex_then_launch()
{
    auto policy = pika::launch::select([]() { return pika::launch::async; });

    pika::future<int> f1 = pika::async(p1);
    pika::future<int> f21 = f1.then(policy, &p2);
    pika::future<int> f2 = f21.then(policy, &p2);
    PIKA_TEST_EQ(f2.get(), 4);
}

///////////////////////////////////////////////////////////////////////////////
void test_complex_then_chain_one()
{
    pika::future<int> f1 = pika::async(p1);
    pika::future<int> f2 = f1.then(&p2).then(&p2);
    PIKA_TEST(f2.get() == 4);
}

void test_complex_then_chain_one_launch()
{
    std::atomic<int> count(0);
    auto policy = pika::launch::select([&count]() -> pika::launch {
        if (count++ == 0)
            return pika::launch::async;
        return pika::launch::sync;
    });

    pika::future<int> f1 = pika::async(p1);
    pika::future<int> f2 = f1.then(policy, &p2).then(policy, &p2);
    PIKA_TEST(f2.get() == 4);
}

///////////////////////////////////////////////////////////////////////////////
void test_complex_then_chain_two()
{
    pika::future<int> f2 = pika::async(p1).then(&p2).then(&p2);
    PIKA_TEST(f2.get() == 4);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    {
        test_return_int();
        test_return_int_launch();
        test_return_void();
        test_return_void_launch();
        test_implicit_unwrapping();
        test_simple_then();
        test_simple_deferred_then();
        test_complex_then();
        test_complex_then_launch();
        test_complex_then_chain_one();
        test_complex_then_chain_one_launch();
        test_complex_then_chain_two();
    }

    pika::local::finalize();
    return pika::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
