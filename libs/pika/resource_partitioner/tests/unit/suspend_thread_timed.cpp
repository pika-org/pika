//  Copyright (c) 2017 Thomas Heller
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
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/testing.hpp>
#include <pika/threading_base/scheduler_mode.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

std::size_t const max_threads = (std::min)(
    std::size_t(4), std::size_t(pika::threads::hardware_concurrency()));

int pika_main(int argc, char* argv[])
{
    pika::threads::thread_pool_base& worker_pool =
        pika::resource::get_thread_pool("default");
    std::cout << "Starting test with scheduler "
              << worker_pool.get_scheduler()->get_description() << std::endl;
    std::size_t const num_threads = pika::resource::get_num_threads("default");

    PIKA_TEST_EQ(max_threads, num_threads);

    pika::threads::thread_pool_base& tp =
        pika::resource::get_thread_pool("default");

    {
        // Check random scheduling with reducing resources.
        std::size_t thread_num = 0;
        bool up = true;
        std::vector<pika::future<void>> fs;

        pika::execution::parallel_executor exec(
            pika::resource::get_thread_pool("default"));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1, 100);

        pika::chrono::high_resolution_timer t;

        while (t.elapsed() < 1)
        {
            for (std::size_t i = 0;
                 i < pika::resource::get_num_threads("default"); ++i)
            {
                fs.push_back(pika::parallel::execution::async_execute_after(
                    exec, std::chrono::milliseconds(dist(gen)), []() {}));
            }

            if (up)
            {
                if (thread_num < pika::resource::get_num_threads("default") - 1)
                {
                    pika::threads::suspend_processing_unit(tp, thread_num).get();
                }

                ++thread_num;

                if (thread_num == pika::resource::get_num_threads("default"))
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
    init_args.rp_callback = [scheduler](auto& rp) {
        std::cout << "\nCreating pool with scheduler " << scheduler
                  << std::endl;

        rp.create_thread_pool("default", scheduler,
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::default_mode |
                pika::threads::policies::enable_elasticity));
    };

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
}

int main(int argc, char* argv[])
{
    PIKA_ASSERT(max_threads >= 2);

    // NOTE: Static schedulers do not support suspending the own worker thread
    // because they do not steal work. Periodic priority scheduler not tested
    // because it does not take into account scheduler states when scheduling
    // work.

    {
        // These schedulers should succeed
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
        };

        for (auto const scheduler : schedulers)
        {
            test_scheduler(argc, argv, scheduler);
        }
    }

    {
        // These schedulers should fail
        std::vector<pika::resource::scheduling_policy> schedulers = {
            pika::resource::scheduling_policy::static_,
            pika::resource::scheduling_policy::static_priority,
            // until timed thread problems are fix, disable this
            //pika::resource::scheduling_policy::shared_priority,
        };

        for (auto const scheduler : schedulers)
        {
            bool exception_thrown = false;
            try
            {
                test_scheduler(argc, argv, scheduler);
            }
            catch (pika::exception const&)
            {
                exception_thrown = true;
            }

            PIKA_TEST(exception_thrown);
        }
    }

    return pika::util::report_errors();
}
