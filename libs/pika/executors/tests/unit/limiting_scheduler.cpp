//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/executors/limiting_scheduler.hpp>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

// this test launches many tasks continuously using a limiting scheduler
// some of the tasks will suspend themselves randomly and so new tasks will be
// spawned until the max tasks 'in flight' limit is reached. Once this happens
// no new tasks should be created and the number 'in flight' should remain
// below/equal to the limit. We track this using some counters and check that
// at the end of the test, the max counter never exceeded the limit.

// type for counters we will use to track tasks in flight
using atomic_type = std::atomic<std::int64_t>;

namespace ex = pika::execution::experimental;
using namespace pika::debug::detail;

///////////////////////////////////////////////////////////////////////////////
/// \brief update_maximum atomically using CAS
/// \param atomic maximum_value
/// \param new value
///
template <typename T>
void update_maximum(std::atomic<T>& maximum, T const& value) noexcept
{
    // if another thread changes the max, then this will either retry, and
    // then succeed in updating, or fail if the other thread set a higher value
    // (because on fail, prev is updated by CAS)
    T prev = maximum;
    while ((value > prev) && !maximum.compare_exchange_weak(prev, value))
    {
    }
}

///////////////////////////////////////////////////////////////////////////////
// random work
void work_fn()
{
    std::default_random_engine eng;
    std::uniform_int_distribution<std::size_t> idist(10, 50);
    std::size_t loop = idist(eng);
    for (std::size_t i = 0; i < loop * 1000; ++i)
    {
        volatile double x;
        x = std::log10(i);
    }
}

///////////////////////////////////////////////////////////////////////////////
//  simple task that can yield at random and increments counters
void test_fn(atomic_type& active_count, atomic_type& total_count, atomic_type& active_max)
{
    // Increment active task count and max active tasks atomically
    update_maximum(active_max, ++active_count);
    ++total_count;

    // yield some random amount of times to make the test more realistic
    // this allows other tasks to run and tests if the limiting scheduler
    // is doing its job
    std::default_random_engine eng;
    std::uniform_int_distribution<std::size_t> idist(10, 20);
    std::size_t loop = idist(eng);
    for (std::size_t i = 0; i < loop; ++i)
    {
        pika::this_thread::yield();
        work_fn();
    }

    // task is completing, decrement active task count
    --active_count;
}

///////////////////////////////////////////////////////////////////////////////
void test_limit_simple()
{
    const int max_simultaneous_tasks_1 = 32;
    const int max_simultaneous_tasks_2 = 21;
    atomic_type task_active_1(0), task_active_2(0);
    atomic_type task_total_1(0), task_total_2(0);
    atomic_type task_max_1(0), task_max_2(0);
    //
    ex::thread_pool_scheduler sched_1{};
    ex::limiting_scheduler<decltype(sched_1)> limit_1(max_simultaneous_tasks_1, sched_1);
    //
    ex::thread_pool_scheduler sched_2{};
    ex::limiting_scheduler<decltype(sched_2)> limit_2(max_simultaneous_tasks_2, sched_2);
    //
    auto test_lambda = [&]() {
        // schedule a simple task on the limiting scheduler
        auto begin = ex::schedule(limit_1);
        auto work1 = ex::then(begin, [&]() {
            pika::thread::id current_id = pika::this_thread::get_id();
            test_fn(task_active_1, task_total_1, task_max_1);
            return current_id;
        });
        // attach a continuation and check thread id is the same
        auto work2 = ex::then(work1, [](pika::thread::id prev_id) {
            pika::thread::id current_id = pika::this_thread::get_id();
            PIKA_TEST_EQ(prev_id, current_id);
            return current_id;
        });

        // transfer the task onto a second limiting scheduler with fewer max tasks
        auto transfer1 = ex::transfer(work2, limit_2);
        auto work3 = ex::then(transfer1, [&](pika::thread::id prev_id) {
            pika::thread::id current_id = pika::this_thread::get_id();
            PIKA_TEST_NEQ(prev_id, current_id);
            test_fn(task_active_2, task_total_2, task_max_2);
            return current_id;
        });

        // launch the work
        ex::start_detached(std::move(work3));
    };

    // run this loop for N seconds and launch as many tasks as we can
    // then check that there were never more than N active at once
    auto start = std::chrono::steady_clock::now();
    bool ok = true;
    while (ok)
    {
        test_lambda();
        ok = (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500));
    }
    const debug_out = 5;
    ex::lsc_debug<debug_out>.debug(str<>("End of task loop :1"), "active", task_active_1, "total",
        task_total_1, "max", task_max_1);
    ex::lsc_debug<debug_out>.debug(str<>("End of task loop :2"), "active", task_active_2, "total",
        task_total_2, "max", task_max_2);

    // some tasks are still in flight, now yield until all are complete
    // if active tasks are not zero afterwards, this will hang anyway ...
    while (task_active_1 > 0 || task_active_2 > 0)
    {
        pika::this_thread::yield();
    }
    ex::lsc_debug<debug_out>.debug(str<>("End of wait loop :1"), "active", task_active_1, "total",
        task_total_1, "max", task_max_1);
    ex::lsc_debug<debug_out>.debug(str<>("End of wait loop :2"), "active", task_active_2, "total",
        task_total_2, "max", task_max_2);

    PIKA_TEST_LTE(task_max_1, max_simultaneous_tasks_1);
    PIKA_TEST_LTE(task_max_2, max_simultaneous_tasks_2);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_limit_simple();

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=cores"};

    // Initialize and run pika
    pika::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(
        pika::init(pika_main, argc, argv, init_args), 0, "pika main exited with non-zero status");

    return 0;
}
