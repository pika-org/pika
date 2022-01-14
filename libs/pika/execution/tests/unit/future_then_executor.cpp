//  Copyright (C) 2012-2013 Vicente Botet
//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2015-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

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
template <typename Executor>
void test_return_int(Executor& exec)
{
    pika::future<int> f1 = pika::async(exec, &p1);
    PIKA_TEST(f1.valid());
    pika::future<int> f2 = f1.then(exec, &p2);
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
template <typename Executor>
void test_return_void(Executor& exec)
{
    pika::future<int> f1 = pika::async(exec, &p1);
    PIKA_TEST(f1.valid());
    pika::future<void> f2 = f1.then(exec, &p3);
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
template <typename Executor>
void test_implicit_unwrapping(Executor& exec)
{
    pika::future<int> f1 = pika::async(exec, &p1);
    PIKA_TEST(f1.valid());
    pika::future<int> f2 = f1.then(exec, &p4);
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
template <typename Executor>
void test_simple_then(Executor& exec)
{
    pika::future<int> f2 = pika::async(exec, p1).then(exec, &p2);
    PIKA_TEST_EQ(f2.get(), 2);
}

template <typename Executor>
void test_simple_deferred_then(Executor& exec)
{
    pika::future<int> f2 = pika::async(exec, p1).then(exec, &p2);
    PIKA_TEST_EQ(f2.get(), 2);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then(Executor& exec)
{
    pika::future<int> f1 = pika::async(exec, p1);
    pika::future<int> f21 = f1.then(exec, &p2);
    pika::future<int> f2 = f21.then(exec, &p2);
    PIKA_TEST_EQ(f2.get(), 4);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then_chain_one(Executor& exec)
{
    pika::future<int> f1 = pika::async(exec, p1);
    pika::future<int> f2 = f1.then(exec, &p2).then(exec, &p2);
    PIKA_TEST_EQ(f2.get(), 4);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then_chain_two(Executor& exec)
{
    pika::future<int> f2 = pika::async(exec, p1).then(exec, &p2).then(exec, &p2);
    PIKA_TEST(f2.get() == 4);
}

template <typename Executor>
void test_then(Executor& exec)
{
    test_return_int(exec);
    test_return_void(exec);
    test_implicit_unwrapping(exec);
    test_simple_then(exec);
    test_simple_deferred_then(exec);
    test_complex_then(exec);
    test_complex_then_chain_one(exec);
    test_complex_then_chain_two(exec);
}

///////////////////////////////////////////////////////////////////////////////
using pika::program_options::options_description;
using pika::program_options::variables_map;

int pika_main(variables_map&)
{
    {
        pika::execution::sequenced_executor exec;
        test_then(exec);
    }

    {
        pika::execution::parallel_executor exec;
        test_then(exec);
    }

    pika::local::finalize();
    return pika::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " PIKA_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
