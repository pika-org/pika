//  Copyright (c)      2019 Mikael Simberg
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The parallel_executor has a constructor that takes a thread_pool_base as an
// argument and executes all its work on that thread pool. This checks that the
// usual functions of an executor work with this executor when used *without the
// pika runtime*. This test fails if thread pools, schedulers etc. assume that
// the global runtime (configuration, thread manager, etc.) always exists.

#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/thread_pools.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
pika::thread::id test(int passed_through)
{
    PIKA_TEST_EQ(passed_through, 42);
    return pika::this_thread::get_id();
}

template <typename Executor>
void test_sync(Executor& exec)
{
    PIKA_TEST(pika::parallel::execution::sync_execute(exec, &test, 42) ==
        pika::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
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

template <typename Executor>
void test_then(Executor& exec)
{
    pika::future<void> f = pika::make_ready_future();

    PIKA_TEST(
        pika::parallel::execution::then_execute(exec, &test_f, f, 42).get() !=
        pika::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, pika::thread::id tid, int passed_through)    //-V813
{
    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
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
        exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42);
    pika::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, pika::util::bind(&bulk_test, _1, tid, _2), v, 42))
        .get();
    pika::when_all(pika::parallel::execution::bulk_async_execute(
                      exec, &bulk_test, v, tid, 42))
        .get();
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int, pika::shared_future<void> f, pika::thread::id tid,
    int passed_through)    //-V813
{
    PIKA_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    PIKA_TEST_NEQ(tid, pika::this_thread::get_id());
    PIKA_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_then(Executor& exec)
{
    pika::thread::id tid = pika::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;
    using pika::util::placeholders::_3;

    pika::shared_future<void> f = pika::make_ready_future();

    pika::parallel::execution::bulk_then_execute(
        exec, pika::util::bind(&bulk_test_f, _1, _2, tid, _3), v, f, 42)
        .get();
    pika::parallel::execution::bulk_then_execute(
        exec, &bulk_test_f, v, f, tid, 42)
        .get();
}

///////////////////////////////////////////////////////////////////////////////
void test_thread_pool_os_executor(pika::execution::parallel_executor exec)
{
    test_sync(exec);
    test_async(exec);
    test_then(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
    test_bulk_then(exec);
}

int main()
{
    {
        // Choose a scheduler.
        using sched_type =
            pika::threads::policies::local_priority_queue_scheduler<>;

        // Choose all the parameters for the thread pool and scheduler.
        std::size_t const num_threads = (std::min)(
            std::size_t(4), std::size_t(pika::threads::hardware_concurrency()));
        std::size_t const max_cores = num_threads;
        pika::threads::policies::detail::affinity_data ad{};
        ad.init(num_threads, max_cores, 0, 1, 0, "core", "balanced", true);
        pika::threads::policies::callback_notifier notifier{};
        pika::threads::policies::thread_queue_init_parameters
            thread_queue_init{};
        sched_type::init_parameter_type scheduler_init(
            num_threads, ad, num_threads, thread_queue_init, "my_scheduler");
        pika::threads::detail::network_background_callback_type
            network_callback{};
        pika::threads::thread_pool_init_parameters thread_pool_init("my_pool", 0,
            pika::threads::policies::scheduler_mode::default_mode, num_threads,
            0, notifier, ad, network_callback, 0,
            (std::numeric_limits<std::int64_t>::max)(),
            (std::numeric_limits<std::int64_t>::max)());

        // Create the scheduler, thread pool, and associated executor.
        std::unique_ptr<sched_type> scheduler{new sched_type(scheduler_init)};
        pika::threads::detail::scheduled_thread_pool<sched_type> pool{
            std::move(scheduler), thread_pool_init};
        pika::execution::parallel_executor exec{&pool};

        // Run the pool.
        std::mutex m;
        std::unique_lock<std::mutex> l(m);
        pool.run(l, num_threads);

        // We can't wait for futures on the main thread, so we spawn a thread to
        // run the tests for us.
        pika::apply(exec, &test_thread_pool_os_executor, exec);

        // Stop the pool.
        pool.stop(l, true);
    }

    return pika::util::report_errors();
}
