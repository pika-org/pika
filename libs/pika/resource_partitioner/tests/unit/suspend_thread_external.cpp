//  Copyright (c) 2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/assert.hpp>
#include <pika/local/chrono.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/testing.hpp>
#include <pika/thread_pool_util/thread_pool_suspension_helpers.hpp>
#include <pika/threading_base/scheduler_mode.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::size_t const max_threads = (std::min)(
    std::size_t(4), std::size_t(pika::threads::hardware_concurrency()));

int pika_main()
{
    std::size_t const num_threads = pika::resource::get_num_threads("worker");

    PIKA_TEST_EQ(std::size_t(max_threads - 1), num_threads);

    pika::threads::thread_pool_base& tp =
        pika::resource::get_thread_pool("worker");

    {
        // Check number of used resources
        for (std::size_t thread_num = 0; thread_num < num_threads - 1;
             ++thread_num)
        {
            pika::threads::suspend_processing_unit(tp, thread_num).get();
            PIKA_TEST_EQ(std::size_t(num_threads - thread_num - 1),
                tp.get_active_os_thread_count());
        }

        for (std::size_t thread_num = 0; thread_num < num_threads - 1;
             ++thread_num)
        {
            pika::threads::resume_processing_unit(tp, thread_num).get();
            PIKA_TEST_EQ(
                std::size_t(thread_num + 2), tp.get_active_os_thread_count());
        }
    }

    {
        // Check suspending and resuming the same thread without waiting for
        // each to finish.
        for (std::size_t thread_num = 0;
             thread_num < pika::resource::get_num_threads("worker");
             ++thread_num)
        {
            std::vector<pika::future<void>> fs;

            fs.push_back(pika::threads::suspend_processing_unit(tp, thread_num));
            fs.push_back(pika::threads::resume_processing_unit(tp, thread_num));

            pika::wait_all(fs);

            // Suspend is not guaranteed to run before resume, so make sure
            // processing unit is running
            pika::threads::resume_processing_unit(tp, thread_num).get();

            fs.clear();

            // Launching the same number of tasks as worker threads may deadlock
            // as no thread is available to steal from the current thread.
            for (std::size_t i = 0; i < max_threads - 1; ++i)
            {
                fs.push_back(
                    pika::threads::suspend_processing_unit(tp, thread_num));
            }

            pika::wait_all(fs);

            fs.clear();

            // Launching the same number of tasks as worker threads may deadlock
            // as no thread is available to steal from the current thread.
            for (std::size_t i = 0; i < max_threads - 1; ++i)
            {
                fs.push_back(
                    pika::threads::resume_processing_unit(tp, thread_num));
            }

            pika::wait_all(fs);
        }
    }

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<pika::future<void>> fs;
        pika::chrono::high_resolution_timer t;
        while (t.elapsed() < 2)
        {
            for (std::size_t i = 0;
                 i < pika::resource::get_num_threads("worker") * 10; ++i)
            {
                fs.push_back(pika::async([]() {}));
            }

            if (up)
            {
                if (thread_num < pika::resource::get_num_threads("worker"))
                {
                    pika::threads::suspend_processing_unit(tp, thread_num).get();
                }

                ++thread_num;

                if (thread_num == pika::resource::get_num_threads("worker"))
                {
                    up = false;
                    --thread_num;
                }
            }
            else
            {
                pika::threads::resume_processing_unit(tp, thread_num).get();

                if (thread_num > 0)
                {
                    --thread_num;
                }
                else
                {
                    up = true;
                }
            }
        }

        pika::when_all(std::move(fs)).get();

        // Don't exit with suspended pus
        for (std::size_t thread_num_resume = 0; thread_num_resume < thread_num;
             ++thread_num_resume)
        {
            pika::threads::resume_processing_unit(tp, thread_num_resume).get();
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
        rp.create_thread_pool("worker", scheduler,
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::default_mode |
                pika::threads::policies::enable_elasticity));

        std::size_t const worker_pool_threads = max_threads - 1;
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
