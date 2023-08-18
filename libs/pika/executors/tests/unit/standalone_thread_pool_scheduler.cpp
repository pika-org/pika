//  Copyright (c)      2019 Mikael Simberg
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The thread_pool_scheduler has a constructor that takes a thread_pool_base as an argument and
// executes all its work on that thread pool. This checks that the usual functions of an scheduler
// work with this executor when used *without the pika runtime*. This test fails if thread pools,
// schedulers etc. assume that the global runtime (configuration, thread manager, etc.) always
// exists.

#include <pika/execution.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/thread_pools.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;
void test_thread_pool_scheduler(pika::execution::experimental::thread_pool_scheduler sched)
{
    auto id =
        tt::sync_wait(ex::schedule(sched) | ex::then([] { return pika::this_thread::get_id(); }));
    PIKA_TEST_NEQ(id, pika::this_thread::get_id());
    PIKA_TEST_NEQ(pika::thread::id(), pika::this_thread::get_id());
}

int main()
{
    {
        // Choose a scheduler.
        using sched_type = pika::threads::detail::local_priority_queue_scheduler<>;

        // Choose all the parameters for the thread pool and scheduler.
        std::size_t const num_threads =
            (std::min)(std::size_t(4), std::size_t(pika::threads::detail::hardware_concurrency()));
        std::size_t const max_cores = num_threads;
        pika::detail::affinity_data ad{};
        ad.init(num_threads, max_cores, 0, 1, 0, "core", "balanced", true);
        pika::threads::callback_notifier notifier{};
        pika::threads::detail::thread_queue_init_parameters thread_queue_init{};
        sched_type::init_parameter_type scheduler_init(
            num_threads, ad, num_threads, thread_queue_init, "my_scheduler");
        pika::threads::detail::thread_pool_init_parameters thread_pool_init("my_pool", 0,
            pika::threads::scheduler_mode::default_mode, num_threads, 0, notifier, ad,
            (std::numeric_limits<std::int64_t>::max)(), (std::numeric_limits<std::int64_t>::max)());

        // Create the scheduler, thread pool, and P2300 scheduler.
        std::unique_ptr<sched_type> scheduler{new sched_type(scheduler_init)};
        pika::threads::detail::scheduled_thread_pool<sched_type> pool{
            std::move(scheduler), thread_pool_init};
        pika::execution::experimental::thread_pool_scheduler sched{&pool};

        // Run the pool.
        std::mutex m;
        std::unique_lock<std::mutex> l(m);
        pool.run(l, num_threads);

        tt::sync_wait(ex::schedule(sched) |
            ex::then(pika::util::detail::bind_front(&test_thread_pool_scheduler, sched)));

        // Stop the pool.
        pool.stop(l, true);
    }

    return 0;
}
