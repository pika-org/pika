//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/assert.hpp>
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
#include <string>
#include <utility>
#include <vector>

std::size_t const max_threads = (std::min)(
    std::size_t(4), std::size_t(pika::threads::hardware_concurrency()));

int pika_main()
{
    std::size_t const num_threads = pika::resource::get_num_threads("default");

    PIKA_TEST_EQ(std::size_t(max_threads), num_threads);

    pika::threads::thread_pool_base& tp =
        pika::resource::get_thread_pool("default");

    PIKA_TEST_EQ(tp.get_active_os_thread_count(), std::size_t(max_threads));

    // Remove all but one pu
    for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
    {
        pika::threads::suspend_processing_unit(tp, thread_num).get();
    }

    // Schedule some dummy work
    for (std::size_t i = 0; i < 10000; ++i)
    {
        pika::apply([]() {});
    }

    // Start shutdown
    return pika::local::finalize();
}

void test_scheduler(
    int argc, char* argv[], pika::resource::scheduling_policy scheduler)
{
    pika::local::init_params init_args;

    init_args.cfg = {"pika.os_threads=" + std::to_string(max_threads)};
    init_args.rp_callback = [scheduler](auto& rp,
                                pika::program_options::variables_map const&) {
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
            // The shared priority scheduler may choose not to create a thread,
            // even when run_now = true and a thread is expected. This can fire
            // an assert in the scheduling_loop if a background thread is not
            // created.
            //pika::resource::scheduling_policy::shared_priority,
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
