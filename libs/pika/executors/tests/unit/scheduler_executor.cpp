//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/latch.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
bool executed = false;

void test_post_f(int passed_through, pika::lcos::local::latch& l)
{
    PIKA_TEST_EQ(passed_through, 42);

    executed = true;

    l.count_down(1);
}

template <typename Executor>
void test_post(Executor&& exec)
{
    executed = false;

    pika::lcos::local::latch l(2);
    pika::parallel::execution::post(exec, &test_post_f, 42, std::ref(l));
    l.arrive_and_wait();

    PIKA_TEST(executed);
}

///////////////////////////////////////////////////////////////////////////////
void test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);

    executed = true;
}

template <typename Executor>
void test_sync(Executor&& exec)
{
    executed = false;

    pika::parallel::execution::sync_execute(exec, &test, 42);

    PIKA_TEST(executed);
}

template <typename Executor>
void test_async(Executor&& exec)
{
    executed = false;

    pika::parallel::execution::async_execute(exec, &test, 42).get();

    PIKA_TEST(executed);
}

///////////////////////////////////////////////////////////////////////////////
void test_f(pika::future<void> f, int passed_through)
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);

    executed = true;
}

template <typename Executor>
void test_then(Executor&& exec)
{
    pika::future<void> f = pika::make_ready_future();

    executed = false;

    pika::parallel::execution::then_execute(exec, &test_f, std::move(f), 42)
        .get();

    PIKA_TEST(executed);
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_void(int seq, int passed_through)    //-V813
{
    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        executed = true;
    }
}

int bulk_test(int seq, int passed_through)    //-V813
{
    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        executed = true;
    }

    return seq;
}

template <typename Executor>
void test_bulk_sync_void(Executor&& exec)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executed = false;

    pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&bulk_test, _1, _2), 107, 42);

    PIKA_TEST(executed);

    executed = false;

    pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, 107, 42);

    PIKA_TEST(executed);
}

template <typename Executor>
void test_bulk_async_void(Executor&& exec)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executed = false;

    auto result = pika::parallel::execution::bulk_async_execute(
        exec, pika::util::bind(&bulk_test, _1, _2), 107, 42);
    pika::when_all(std::move(result)).get();

    PIKA_TEST(executed);

    executed = false;

    pika::when_all(
        pika::parallel::execution::bulk_async_execute(exec, &bulk_test, 107, 42))
        .get();

    PIKA_TEST(executed);
}

template <typename Executor>
void test_bulk_async(Executor&& exec)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executed = false;
    int const n = 107;

    auto fut_result = pika::parallel::execution::bulk_async_execute(
        exec, pika::util::bind(&bulk_test, _1, _2), n, 42);
    auto result = pika::when_all(std::move(fut_result)).get();

    for (int i = 0; i < n; ++i)
    {
        PIKA_TEST_EQ(i, result[i].get());
    }

    PIKA_TEST(executed);

    executed = false;

    pika::when_all(
        pika::parallel::execution::bulk_async_execute(exec, &bulk_test, 107, 42))
        .get();

    PIKA_TEST(executed);
}

///////////////////////////////////////////////////////////////////////////////
int bulk_test_f(int seq, pika::shared_future<void> f,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        executed = true;
    }

    return seq;
}

template <typename Executor>
void test_bulk_then(Executor&& exec)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    {
        executed = false;

        auto result = pika::parallel::execution::bulk_then_execute(
            exec, pika::util::bind(&bulk_test_f, _1, _2, _3), 107, f, 42)
                          .get();

        PIKA_TEST(executed);
        PIKA_TEST(result.size() == 107);

        int expected = 0;
        for (auto i : result)
        {
            PIKA_TEST_EQ(i, expected);
            ++expected;
        }
    }

    {
        executed = false;

        auto result = pika::parallel::execution::bulk_then_execute(
            exec, &bulk_test_f, 107, f, 42)
                          .get();

        PIKA_TEST(executed);
        PIKA_TEST(result.size() == 107);

        int expected = 0;
        for (auto i : result)
        {
            PIKA_TEST_EQ(i, expected);
            ++expected;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f_void(int seq, pika::shared_future<void> f,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        executed = true;
    }
}

template <typename Executor>
void test_bulk_then_void(Executor&& exec)
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    executed = false;

    pika::parallel::execution::bulk_then_execute(
        exec, pika::util::bind(&bulk_test_f_void, _1, _2, _3), 107, f, 42)
        .get();

    PIKA_TEST(executed);

    executed = false;

    pika::parallel::execution::bulk_then_execute(
        exec, &bulk_test_f_void, 107, f, 42)
        .get();

    PIKA_TEST(executed);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_executor(Executor&& exec)
{
    test_post(exec);

    test_sync(exec);
    test_async(exec);
    test_then(exec);

    test_bulk_sync_void(exec);
    test_bulk_async_void(exec);
    test_bulk_async(exec);
    test_bulk_then(exec);
    test_bulk_then_void(exec);
}

int pika_main()
{
    using namespace pika::execution::experimental;

    scheduler_executor exec(thread_pool_scheduler{});

    test_executor(exec);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
