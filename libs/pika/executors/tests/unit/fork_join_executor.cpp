//  Copyright (c)      2020 ETH Zurich
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/chrono.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using pika::execution::experimental::fork_join_executor;

static std::atomic<std::size_t> count{0};

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, int passed_through)    //-V813
{
    ++count;
    PIKA_TEST_EQ(passed_through, 42);
}

template <typename... ExecutorArgs>
void test_bulk_sync(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_sync\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    pika::parallel::execution::bulk_sync_execute(
        exec, pika::util::bind(&bulk_test, _1, _2), v, 42);
    PIKA_TEST_EQ(count.load(), n);

    pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, 42);
    PIKA_TEST_EQ(count.load(), 2 * n);
}

template <typename... ExecutorArgs>
void test_bulk_async(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_async\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, pika::util::bind(&bulk_test, _1, _2), v, 42))
        .get();
    PIKA_TEST_EQ(count.load(), n);

    pika::when_all(
        pika::parallel::execution::bulk_async_execute(exec, &bulk_test, v, 42))
        .get();
    PIKA_TEST_EQ(count.load(), 2 * n);
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_exception(int, int passed_through)    //-V813
{
    PIKA_TEST_EQ(passed_through, 42);
    throw std::runtime_error("test");
}

template <typename... ExecutorArgs>
void test_bulk_sync_exception(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_sync_exception\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    bool caught_exception = false;
    try
    {
        pika::parallel::execution::bulk_sync_execute(
            exec, &bulk_test_exception, v, 42);

        PIKA_TEST(false);
    }
    catch (std::runtime_error const& /*e*/)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename... ExecutorArgs>
void test_bulk_async_exception(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_async_exception\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    bool caught_exception = false;
    try
    {
        auto r = pika::parallel::execution::bulk_async_execute(
            exec, &bulk_test_exception, v, 42);
        PIKA_TEST_EQ(r.size(), std::size_t(1));
        r[0].get();

        PIKA_TEST(false);
    }
    catch (std::runtime_error const& /*e*/)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

void static_check_executor()
{
    using namespace pika::traits;

    static_assert(!has_sync_execute_member<fork_join_executor>::value,
        "!has_sync_execute_member<fork_join_executor>::value");
    static_assert(!has_async_execute_member<fork_join_executor>::value,
        "!has_async_execute_member<fork_join_executor>::value");
    static_assert(!has_then_execute_member<fork_join_executor>::value,
        "!has_then_execute_member<fork_join_executor>::value");
    static_assert(has_bulk_sync_execute_member<fork_join_executor>::value,
        "has_bulk_sync_execute_member<fork_join_executor>::value");
    static_assert(has_bulk_async_execute_member<fork_join_executor>::value,
        "has_bulk_async_execute_member<fork_join_executor>::value");
    static_assert(!has_bulk_then_execute_member<fork_join_executor>::value,
        "!has_bulk_then_execute_member<fork_join_executor>::value");
    static_assert(!has_post_member<fork_join_executor>::value,
        "!has_post_member<fork_join_executor>::value");
}

template <typename... ExecutorArgs>
void test_executor(pika::threads::thread_priority priority,
    pika::threads::thread_stacksize stacksize,
    fork_join_executor::loop_schedule schedule)
{
    std::cerr << "testing fork_join_executor with priority = " << priority
              << ", stacksize = " << stacksize << ", schedule = " << schedule
              << "\n";
    test_bulk_sync(priority, stacksize, schedule);
    test_bulk_async(priority, stacksize, schedule);
    test_bulk_sync_exception(priority, stacksize, schedule);
    test_bulk_async_exception(priority, stacksize, schedule);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    static_check_executor();

    // thread_stacksize::nostack cannot be used with the fork_join_executor
    // because it prevents other work from running when yielding. Using
    // thread_priority::low hangs for unknown reasons.
    for (auto const priority : {
             // pika::threads::thread_priority::low,
             pika::threads::thread_priority::normal,
             pika::threads::thread_priority::high,
         })
    {
        for (auto const stacksize : {
                 // pika::threads::thread_stacksize::nostack,
                 pika::threads::thread_stacksize::small_,
             })
        {
            for (auto const schedule : {
                     fork_join_executor::loop_schedule::static_,
                     fork_join_executor::loop_schedule::dynamic,
                 })
            {
                {
                    test_executor(priority, stacksize, schedule);
                }
            }
        }
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
