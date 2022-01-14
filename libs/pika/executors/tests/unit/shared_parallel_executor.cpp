//  Copyright (c) 2007-2016 Hartmut Kaiser
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
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct shared_parallel_executor
{
    template <typename F, typename... Ts>
    pika::shared_future<typename pika::util::invoke_result<F, Ts...>::type>
    async_execute(F&& f, Ts&&... ts)
    {
        return pika::async(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace pika { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<shared_parallel_executor> : std::true_type
    {
    };
}}}    // namespace pika::parallel::execution

///////////////////////////////////////////////////////////////////////////////
pika::thread::id test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);
    return pika::this_thread::get_id();
}

void test_sync()
{
    typedef shared_parallel_executor executor;

    executor exec;
    PIKA_TEST(pika::parallel::execution::sync_execute(exec, &test, 42) !=
        pika::this_thread::get_id());
}

void test_async()
{
    typedef shared_parallel_executor executor;

    executor exec;

    pika::shared_future<pika::thread::id> fut =
        pika::parallel::execution::async_execute(exec, &test, 42);

    PIKA_TEST_NEQ(fut.get(), pika::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

void test_bulk_sync()
{
    typedef shared_parallel_executor executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executor exec;
    pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42);
    pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, tid, 42);
}

void test_bulk_async()
{
    typedef shared_parallel_executor executor;

    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    executor exec;
    std::vector<pika::shared_future<void>> futs =
        pika::parallel::execution::bulk_async_execute(
            exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42);
    pika::when_all(futs).get();

    futs = pika::parallel::execution::bulk_async_execute(
        exec, &bulk_test, v, tid, 42);
    pika::when_all(futs).get();
}

///////////////////////////////////////////////////////////////////////////////
void void_test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);
}

void test_sync_void()
{
    typedef shared_parallel_executor executor;

    executor exec;
    pika::parallel::execution::sync_execute(exec, &void_test, 42);
}

void test_async_void()
{
    typedef shared_parallel_executor executor;

    executor exec;
    pika::shared_future<void> fut =
        pika::parallel::execution::async_execute(exec, &void_test, 42);
    fut.get();
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_sync();
    test_async();
    test_bulk_sync();
    test_bulk_async();

    test_sync_void();
    test_async_void();

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
