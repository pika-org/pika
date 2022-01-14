//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
pika::thread::id async_test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);
    return pika::this_thread::get_id();
}

void apply_test(
    pika::lcos::local::latch& l, pika::thread::id& id, int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);
    id = pika::this_thread::get_id();
    l.count_down(1);
}

void async_bulk_test(int, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_apply(Executor& exec)
{
    pika::lcos::local::latch l(2);
    pika::thread::id id;

    pika::parallel::execution::post(
        exec, &apply_test, std::ref(l), std::ref(id), 42);
    l.count_down_and_wait();

    PIKA_TEST_NEQ(id, pika::this_thread::get_id());
}

template <typename Executor>
void test_sync(Executor& exec)
{
    PIKA_TEST(pika::parallel::execution::sync_execute(exec, &async_test, 42) !=
        pika::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    PIKA_TEST(
        pika::parallel::execution::async_execute(exec, &async_test, 42).get() !=
        pika::this_thread::get_id());
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&async_bulk_test, _1, tid, _2), v, 42);
    pika::parallel::execution::bulk_sync_execute(
        exec, &async_bulk_test, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    pika::when_all(pika::parallel::execution::bulk_async_execute(exec,
                      pika::util::bind(&async_bulk_test, _1, tid, _2), v, 42))
        .get();
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, &async_bulk_test, v, tid, 42))
        .get();
}

std::atomic<std::size_t> count_apply(0);
std::atomic<std::size_t> count_sync(0);
std::atomic<std::size_t> count_async(0);
std::atomic<std::size_t> count_bulk_sync(0);
std::atomic<std::size_t> count_bulk_async(0);

template <typename Executor>
void test_executor(std::array<std::size_t, 5> expected)
{
    typedef typename pika::traits::executor_execution_category<Executor>::type
        execution_category;

    PIKA_TEST((std::is_same<pika::execution::parallel_execution_tag,
        execution_category>::value));

    count_apply.store(0);
    count_sync.store(0);
    count_async.store(0);
    count_bulk_sync.store(0);
    count_bulk_async.store(0);

    Executor exec;

    test_apply(exec);
    test_sync(exec);
    test_async(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);

    PIKA_TEST_EQ(expected[0], count_apply.load());
    PIKA_TEST_EQ(expected[1], count_sync.load());
    PIKA_TEST_EQ(expected[2], count_async.load());
    PIKA_TEST_EQ(expected[3], count_bulk_sync.load());
    PIKA_TEST_EQ(expected[4], count_bulk_async.load());
}

///////////////////////////////////////////////////////////////////////////////
struct test_async_executor1
{
    typedef pika::execution::parallel_execution_tag execution_category;

    template <typename F, typename... Ts>
    static pika::future<typename pika::util::invoke_result<F, Ts...>::type>
    async_execute(F&& f, Ts&&... ts)
    {
        ++count_async;
        return pika::async(
            pika::launch::async, std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor1> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

struct test_async_executor2 : test_async_executor1
{
    typedef pika::execution::parallel_execution_tag execution_category;

    template <typename F, typename... Ts>
    static typename pika::util::detail::invoke_deferred_result<F, Ts...>::type
    sync_execute(F&& f, Ts&&... ts)
    {
        ++count_sync;
        return pika::async(
            pika::launch::async, std::forward<F>(f), std::forward<Ts>(ts)...)
            .get();
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor2> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

struct test_async_executor3 : test_async_executor1
{
    typedef pika::execution::parallel_execution_tag execution_category;

    template <typename F, typename Shape, typename... Ts>
    static void bulk_sync_execute(F f, Shape const& shape, Ts&&... ts)
    {
        ++count_bulk_sync;
        std::vector<pika::future<void>> results;
        for (auto const& elem : shape)
        {
            results.push_back(pika::async(pika::launch::async, f, elem, ts...));
        }
        pika::when_all(results).get();
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor3> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

struct test_async_executor4 : test_async_executor1
{
    typedef pika::execution::parallel_execution_tag execution_category;

    template <typename F, typename Shape, typename... Ts>
    static std::vector<pika::future<void>> bulk_async_execute(
        F f, Shape const& shape, Ts&&... ts)
    {
        ++count_bulk_async;
        std::vector<pika::future<void>> results;
        for (auto const& elem : shape)
        {
            results.push_back(pika::async(pika::launch::async, f, elem, ts...));
        }
        return results;
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor4> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<test_async_executor4> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

struct test_async_executor5 : test_async_executor1
{
    typedef pika::execution::parallel_execution_tag execution_category;

    template <typename F, typename... Ts>
    static void post(F&& f, Ts&&... ts)
    {
        ++count_apply;
        pika::apply(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor5> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

template <typename Executor, typename B1, typename B2, typename B3, typename B4,
    typename B5>
void static_check_executor(B1, B2, B3, B4, B5)
{
    using namespace pika::traits;

    static_assert(has_async_execute_member<Executor>::value == B1::value,
        "check has_async_execute_member<Executor>::value");
    static_assert(has_sync_execute_member<Executor>::value == B2::value,
        "check has_sync_execute_member<Executor>::value");
    static_assert(has_bulk_sync_execute_member<Executor>::value == B3::value,
        "check has_bulk_sync_execute_member<Executor>::value");
    static_assert(has_bulk_async_execute_member<Executor>::value == B4::value,
        "check has_bulk_async_execute_member<Executor>::value");
    static_assert(has_post_member<Executor>::value == B5::value,
        "check has_post_member<Executor>::value");

    static_assert(!has_then_execute_member<Executor>::value,
        "!has_then_execute_member<Executor>::value");
    static_assert(!has_bulk_then_execute_member<Executor>::value,
        "!has_bulk_then_execute_member<Executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::false_type f;
    std::true_type t;

    static_check_executor<test_async_executor1>(t, f, f, f, f);
    static_check_executor<test_async_executor2>(t, t, f, f, f);
    static_check_executor<test_async_executor3>(t, f, t, f, f);
    static_check_executor<test_async_executor4>(t, f, f, t, f);
    static_check_executor<test_async_executor5>(t, f, f, f, t);

    test_executor<test_async_executor1>({{0, 0, 431, 0, 0}});
    test_executor<test_async_executor2>({{0, 1, 430, 0, 0}});
    test_executor<test_async_executor3>({{0, 0, 217, 2, 0}});
    test_executor<test_async_executor4>({{0, 0, 217, 0, 2}});
    test_executor<test_async_executor5>({{1, 0, 430, 0, 0}});

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
