//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
pika::thread::id test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);
    return pika::this_thread::get_id();
}

void test_sync()
{
    pika::execution::sequenced_executor exec;
    PIKA_TEST(pika::parallel::execution::sync_execute(exec, &test, 42) ==
        pika::this_thread::get_id());
}

void test_async()
{
    typedef pika::execution::parallel_executor executor;

    executor exec(pika::launch::fork);
    PIKA_TEST(pika::parallel::execution::async_execute(exec, &test, 42).get() !=
        pika::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
pika::thread::id test_f(pika::future<void> f, int passed_through)
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);
    return pika::this_thread::get_id();
}

void test_then()
{
    typedef pika::execution::sequenced_executor executor;

    pika::future<void> f = pika::make_ready_future();

    executor exec;
    PIKA_TEST(
        pika::parallel::execution::then_execute(exec, &test_f, f, 42).get() ==
        pika::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_EQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

void test_bulk_sync()
{
    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    pika::execution::sequenced_executor exec;
    pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42);
    pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, tid, 42);
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int, pika::shared_future<void> f, pika::thread::id tid,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

void test_bulk_then()
{
    typedef pika::execution::sequenced_executor executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    executor exec;
    pika::parallel::execution::bulk_then_execute(
        exec, pika::util::bind(&bulk_test_f, _1, _2, tid, _3), v, f, 42)
        .get();
    pika::parallel::execution::bulk_then_execute(
        exec, &bulk_test_f, v, f, tid, 42)
        .get();
}

void test_bulk_async()
{
    typedef pika::execution::sequenced_executor executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executor exec;
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42))
        .get();
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, &bulk_test, v, tid, 42))
        .get();
}

void static_check_executor()
{
    using namespace pika::traits;
    using executor = pika::execution::sequenced_executor;

    static_assert(has_sync_execute_member<executor>::value,
        "has_sync_execute_member<executor>::value");
    static_assert(has_async_execute_member<executor>::value,
        "has_async_execute_member<executor>::value");
    static_assert(!has_then_execute_member<executor>::value,
        "!has_then_execute_member<executor>::value");
    static_assert(has_bulk_sync_execute_member<executor>::value,
        "has_bulk_sync_execute_member<executor>::value");
    static_assert(has_bulk_async_execute_member<executor>::value,
        "has_bulk_async_execute_member<executor>::value");
    static_assert(!has_bulk_then_execute_member<executor>::value,
        "!has_bulk_then_execute_member<executor>::value");
    static_assert(has_post_member<executor>::value,
        "check has_post_member<executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    static_check_executor();

    test_sync();
    test_async();
    test_then();

    test_bulk_sync();
    test_bulk_async();
    test_bulk_then();

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
