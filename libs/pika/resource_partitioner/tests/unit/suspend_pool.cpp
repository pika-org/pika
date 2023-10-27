//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/assert.hpp>
#include <pika/chrono.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/semaphore.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;

#if defined(PIKA_HAVE_VERIFY_LOCKS)
inline constexpr std::size_t num_tasks_per_worker_thread = 100;
#else
inline constexpr std::size_t num_tasks_per_worker_thread = 10000;
#endif

std::size_t const max_threads =
    (std::min)(std::size_t(4), std::size_t(pika::threads::detail::hardware_concurrency()));

int pika_main()
{
    bool exception_thrown = false;

    try
    {
        pika::this_thread::get_pool()->suspend_direct();
        PIKA_TEST_MSG(false, "Suspending should not be allowed on own pool");
    }
    catch (pika::exception const&)
    {
        exception_thrown = true;
    }

    PIKA_TEST(exception_thrown);

    pika::threads::detail::thread_pool_base& worker_pool =
        pika::resource::get_thread_pool("worker");
    auto worker_sched = ex::thread_pool_scheduler{&pika::resource::get_thread_pool("worker")};
    std::size_t const worker_pool_threads = pika::resource::get_num_threads("worker");

    {
        // Suspend and resume pool
        pika::chrono::detail::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            const std::size_t n = worker_pool_threads * num_tasks_per_worker_thread;
            std::atomic<std::size_t> num_executed{0};

            for (std::size_t i = 0; i < n; ++i)
            {
                ex::execute(worker_sched, [&] { ++num_executed; });
            }

            worker_pool.suspend_direct();

            // All work should be done when pool has been suspended
            PIKA_TEST_EQ(num_executed.load(), n);

            worker_pool.resume_direct();
        }
    }

    {
        // Suspend pool with some threads already suspended
        pika::chrono::detail::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            for (std::size_t thread_num = 0; thread_num < worker_pool_threads - 1; ++thread_num)
            {
                worker_pool.suspend_processing_unit_direct(thread_num);
            }

            const std::size_t n = worker_pool_threads * num_tasks_per_worker_thread;
            std::atomic<std::size_t> num_executed{0};

            for (std::size_t i = 0; i < n; ++i)
            {
                ex::execute(worker_sched, [&] { ++num_executed; });
            }

            worker_pool.suspend_direct();

            // All work should be done when pool has been suspended
            PIKA_TEST_EQ(num_executed.load(), n);

            worker_pool.resume_direct();
        }
    }

    return pika::finalize();
}

void test_scheduler(int argc, char* argv[], pika::resource::scheduling_policy scheduler)
{
    using ::pika::threads::scheduler_mode;

    pika::init_params init_args;

    init_args.cfg = {"pika.os_threads=" + std::to_string(max_threads)};
    init_args.rp_callback = [scheduler](auto& rp, pika::program_options::variables_map const&) {
        rp.create_thread_pool(
            "worker", scheduler, scheduler_mode::default_mode | scheduler_mode::enable_elasticity);

        std::size_t const worker_pool_threads = max_threads - 1;
        PIKA_ASSERT(worker_pool_threads >= 1);
        std::size_t worker_pool_threads_added = 0;

        for (pika::resource::numa_domain const& d : rp.numa_domains())
        {
            for (pika::resource::core const& c : d.cores())
            {
                for (pika::resource::pu const& p : c.pus())
                {
                    if (worker_pool_threads_added < worker_pool_threads)
                    {
                        rp.add_resource(p, "worker");
                        ++worker_pool_threads_added;
                    }
                }
            }
        }
    };

    PIKA_TEST_EQ(pika::init(pika_main, argc, argv, init_args), 0);
}

int main(int argc, char* argv[])
{
    PIKA_ASSERT(max_threads >= 2);

    std::vector<pika::resource::scheduling_policy> schedulers = {
        pika::resource::scheduling_policy::local,
        pika::resource::scheduling_policy::local_priority_fifo,
#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
        pika::resource::scheduling_policy::local_priority_lifo,
#endif
#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
        pika::resource::scheduling_policy::abp_priority_fifo,
        pika::resource::scheduling_policy::abp_priority_lifo,
#endif
        pika::resource::scheduling_policy::static_,
        pika::resource::scheduling_policy::static_priority,
#if !defined(PIKA_HAVE_VERIFY_LOCKS)
        pika::resource::scheduling_policy::shared_priority,
#endif
    };

    for (auto const scheduler : schedulers) { test_scheduler(argc, argv, scheduler); }

    return 0;
}
