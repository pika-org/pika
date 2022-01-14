//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
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

template <typename Policy>
void test_sync()
{
    typedef pika::execution::parallel_policy_executor<Policy> executor;

    executor exec;
    PIKA_TEST(pika::parallel::execution::sync_execute(exec, &test, 42) ==
        pika::this_thread::get_id());
}

template <typename Policy>
void test_async(bool sync)
{
    typedef pika::execution::parallel_policy_executor<Policy> executor;

    executor exec;
    bool result =
        pika::parallel::execution::async_execute(exec, &test, 42).get() ==
        pika::this_thread::get_id();

    PIKA_TEST_EQ(sync, result);
}

///////////////////////////////////////////////////////////////////////////////
pika::thread::id test_f(pika::future<void> f, int passed_through)
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(passed_through, 42);
    return pika::this_thread::get_id();
}

template <typename Policy>
void test_then(bool sync)
{
    typedef pika::execution::parallel_policy_executor<Policy> executor;

    pika::future<void> f = pika::make_ready_future();

    executor exec;
    bool result =
        pika::parallel::execution::then_execute(exec, &test_f, f, 42).get() ==
        pika::this_thread::get_id();

    PIKA_TEST_EQ(sync, result);
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_s(int, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_EQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

void bulk_test_a(int, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

template <typename Policy>
void test_bulk_sync(bool sync)
{
    typedef pika::execution::parallel_policy_executor<Policy> executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executor exec;
    pika::parallel::execution::bulk_sync_execute(exec,
        pika::util::bind(sync ? &bulk_test_s : &bulk_test_a, _1, tid, _2), v,
        42);
    pika::parallel::execution::bulk_sync_execute(
        exec, sync ? &bulk_test_s : &bulk_test_a, v, tid, 42);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
void test_bulk_async(bool sync)
{
    typedef pika::execution::parallel_policy_executor<Policy> executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executor exec;
    pika::when_all(
        pika::parallel::execution::bulk_async_execute(exec,
            pika::util::bind(sync ? &bulk_test_s : &bulk_test_a, _1, tid, _2), v,
            42))
        .get();
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, sync ? &bulk_test_s : &bulk_test_a, v, tid, 42))
        .get();
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f_s(int, pika::shared_future<void> f, pika::thread::id tid,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_EQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

void bulk_test_f_a(int, pika::shared_future<void> f, pika::thread::id tid,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

template <typename Policy>
void test_bulk_then(bool sync)
{
    typedef pika::execution::parallel_policy_executor<Policy> executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    executor exec;
    pika::parallel::execution::bulk_then_execute(exec,
        pika::util::bind(
            sync ? &bulk_test_f_s : &bulk_test_f_a, _1, _2, tid, _3),
        v, f, 42)
        .get();
    pika::parallel::execution::bulk_then_execute(
        exec, sync ? &bulk_test_f_s : &bulk_test_f_a, v, f, tid, 42)
        .get();
}

template <typename Policy>
void static_check_executor()
{
    using namespace pika::traits;
    using executor = pika::execution::parallel_policy_executor<Policy>;

    static_assert(has_sync_execute_member<executor>::value,
        "has_sync_execute_member<executor>::value");
    static_assert(has_async_execute_member<executor>::value,
        "has_async_execute_member<executor>::value");
    static_assert(has_then_execute_member<executor>::value,
        "has_then_execute_member<executor>::value");
    static_assert(!has_bulk_sync_execute_member<executor>::value,
        "!has_bulk_sync_execute_member<executor>::value");
    static_assert(has_bulk_async_execute_member<executor>::value,
        "has_bulk_async_execute_member<executor>::value");
    static_assert(has_bulk_then_execute_member<executor>::value,
        "has_bulk_then_execute_member<executor>::value");
    static_assert(has_post_member<executor>::value,
        "check has_post_member<executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
void policy_test(bool sync = false)
{
    static_check_executor<Policy>();

    test_sync<Policy>();
    test_async<Policy>(sync);
    test_then<Policy>(sync);

    test_bulk_sync<Policy>(sync);
    test_bulk_async<Policy>(sync);
    test_bulk_then<Policy>(sync);
}

int pika_main()
{
    policy_test<pika::launch>();

    policy_test<pika::launch::async_policy>();
    policy_test<pika::launch::sync_policy>(true);
    policy_test<pika::launch::fork_policy>();
    policy_test<pika::launch::deferred_policy>(true);

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
