//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/bind.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/execution.hpp>
#include <pika/modules/executors.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);
    return passed_through;
}

using executor = pika::parallel::execution::polymorphic_executor<int(int)>;

void test_sync(executor const& exec)
{
    PIKA_TEST(pika::parallel::execution::sync_execute(exec, &test, 42) == 42);
}

void test_async(executor const& exec)
{
    PIKA_TEST(
        pika::parallel::execution::async_execute(exec, &test, 42).get() == 42);
}

///////////////////////////////////////////////////////////////////////////////
int test_f(pika::shared_future<void> const& f, int passed_through)
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);
    return 42;
}

void test_then(executor const& exec)
{
    pika::future<void> f = pika::make_ready_future();

    PIKA_TEST(
        pika::parallel::execution::then_execute(exec, &test_f, std::move(f), 42)
            .get() == 42);
}

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::size_t> count(0);

int bulk_test(std::size_t, int passed_through)
{
    ++count;
    PIKA_TEST_EQ(passed_through, 42);
    return passed_through;
}

void test_bulk_sync(executor const& exec)
{
    std::vector<int> v(107);
    std::iota(v.begin(), v.end(), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    count = 0;
    pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&bulk_test, _1, _2), v, 42);
    PIKA_TEST(count == v.size());

    count = 0;
    pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, 42);
    PIKA_TEST(count == v.size());
}

///////////////////////////////////////////////////////////////////////////////
void test_bulk_async(executor const& exec)
{
    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    count = 0;
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, pika::util::bind(&bulk_test, _1, _2), v, 42))
        .get();
    PIKA_TEST(count == v.size());

    count = 0;
    pika::when_all(
        pika::parallel::execution::bulk_async_execute(exec, &bulk_test, v, 42))
        .get();
    PIKA_TEST(count == v.size());
}

///////////////////////////////////////////////////////////////////////////////
int bulk_test_f(
    std::size_t, pika::shared_future<void> const& f, int passed_through)
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    ++count;
    PIKA_TEST_EQ(passed_through, 42);
    return passed_through;
}

void test_bulk_then(executor const& exec)
{
    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    count = 0;
    pika::parallel::execution::bulk_then_execute(
        exec, pika::util::bind(&bulk_test_f, _1, _2, _3), v, f, 42)
        .get();
    PIKA_TEST(count == v.size());

    count = 0;
    pika::parallel::execution::bulk_then_execute(exec, &bulk_test_f, v, f, 42)
        .get();
    PIKA_TEST(count == v.size());
}

void static_check_executor()
{
    using namespace pika::traits;

    static_assert(has_sync_execute_member<executor>::value,
        "has_sync_execute_member<executor>::value");
    static_assert(has_async_execute_member<executor>::value,
        "has_async_execute_member<executor>::value");
    static_assert(has_then_execute_member<executor>::value,
        "has_then_execute_member<executor>::value");
    static_assert(has_bulk_sync_execute_member<executor>::value,
        "has_bulk_sync_execute_member<executor>::value");
    static_assert(has_bulk_async_execute_member<executor>::value,
        "has_bulk_async_execute_member<executor>::value");
    static_assert(has_bulk_then_execute_member<executor>::value,
        "has_bulk_then_execute_member<executor>::value");
    static_assert(has_post_member<executor>::value,
        "check has_post_member<executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
void test_executor(executor const& exec)
{
    test_sync(exec);
    test_async(exec);
    test_then(exec);

    test_bulk_sync(exec);
    test_bulk_async(exec);
    test_bulk_then(exec);
}

int pika_main()
{
    static_check_executor();

    pika::execution::parallel_executor par_exec;
    test_executor(executor(par_exec));

    pika::execution::sequenced_executor seq_exec;
    test_executor(executor(seq_exec));

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
