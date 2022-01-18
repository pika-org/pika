//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/assert.hpp>
#include <pika/local/chrono.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/semaphore.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threadmanager.hpp>
#include <pika/thread_pool_util/thread_pool_suspension_helpers.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/threading_base/thread_helpers.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

std::size_t const max_threads = (std::min)(
    std::size_t(4), std::size_t(pika::threads::hardware_concurrency()));

int pika_main()
{
    bool exception_thrown = false;

    try
    {
        // Use .get() to throw exception
        pika::threads::suspend_pool(*pika::this_thread::get_pool()).get();
        PIKA_TEST_MSG(false, "Suspending should not be allowed on own pool");
    }
    catch (pika::exception const&)
    {
        exception_thrown = true;
    }

    PIKA_TEST(exception_thrown);

    pika::threads::thread_pool_base& worker_pool =
        pika::resource::get_thread_pool("worker");
    pika::execution::parallel_executor worker_exec(
        &pika::resource::get_thread_pool("worker"));
    std::size_t const worker_pool_threads =
        pika::resource::get_num_threads("worker");

    {
        // Suspend and resume pool with future
        pika::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            std::vector<pika::future<void>> fs;

            for (std::size_t i = 0; i < worker_pool_threads * 10000; ++i)
            {
                fs.push_back(pika::async(worker_exec, []() {}));
            }

            pika::threads::suspend_pool(worker_pool).get();

            // All work should be done when pool has been suspended
            PIKA_TEST(pika::when_all(std::move(fs)).is_ready());

            pika::threads::resume_pool(worker_pool).get();
        }
    }

    {
        // Suspend and resume pool with callback
        pika::lcos::local::counting_semaphore sem;
        pika::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            std::vector<pika::future<void>> fs;

            for (std::size_t i = 0; i < worker_pool_threads * 10000; ++i)
            {
                fs.push_back(pika::async(worker_exec, []() {}));
            }

            pika::threads::suspend_pool_cb(
                worker_pool, [&sem]() { sem.signal(); });

            sem.wait(1);

            // All work should be done when pool has been suspended
            PIKA_TEST(pika::when_all(std::move(fs)).is_ready());

            pika::threads::resume_pool_cb(
                worker_pool, [&sem]() { sem.signal(); });

            sem.wait(1);
        }
    }

    {
        // Suspend pool with some threads already suspended
        pika::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            for (std::size_t thread_num = 0;
                 thread_num < worker_pool_threads - 1; ++thread_num)
            {
                pika::threads::suspend_processing_unit(worker_pool, thread_num);
            }

            std::vector<pika::future<void>> fs;

            for (std::size_t i = 0;
                 i < pika::resource::get_num_threads("default") * 10000; ++i)
            {
                fs.push_back(pika::async(worker_exec, []() {}));
            }

            pika::threads::suspend_pool(worker_pool).get();

            // All work should be done when pool has been suspended
            PIKA_TEST(pika::when_all(std::move(fs)).is_ready());

            pika::threads::resume_pool(worker_pool).get();
        }
    }

    return pika::local::finalize();
}

void test_scheduler(
    int argc, char* argv[], pika::resource::scheduling_policy scheduler)
{
    pika::local::init_params init_args;

    init_args.cfg = {"pika.os_threads=" + std::to_string(max_threads)};
    init_args.rp_callback = [scheduler](auto& rp,
                                pika::program_options::variables_map const&) {
        rp.create_thread_pool("worker", scheduler);

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

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
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
        pika::resource::scheduling_policy::shared_priority,
    };

    for (auto const scheduler : schedulers)
    {
        test_scheduler(argc, argv, scheduler);
    }

    return pika::util::report_errors();
}
