//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/testing.hpp>
#include <pika/threading_base/scheduler_mode.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

int pika_main()
{
    bool run = false;
    pika::future<void> f1 = pika::async([&run]() { run = true; });

    if (!run)
    {
        // This thread should get scheduled last (because of
        // pika::threads::thread_schedule_state::pending) and let the function
        // spawned above run.
        pika::this_thread::suspend(pika::threads::thread_schedule_state::pending);
    }

    PIKA_TEST(run);

    return pika::local::finalize();
}

template <typename Scheduler>
void test_scheduler(int argc, char* argv[])
{
    pika::local::init_params init_args;

    init_args.cfg = {"pika.os_threads=1"};
    init_args.rp_callback = [](auto& rp,
                                pika::program_options::variables_map const&) {
        rp.create_thread_pool("default",
            [](pika::threads::thread_pool_init_parameters thread_pool_init,
                pika::threads::policies::thread_queue_init_parameters
                    thread_queue_init)
                -> std::unique_ptr<pika::threads::thread_pool_base> {
                typename Scheduler::init_parameter_type init(
                    thread_pool_init.num_threads_,
                    thread_pool_init.affinity_data_, std::size_t(-1),
                    thread_queue_init);
                std::unique_ptr<Scheduler> scheduler(new Scheduler(init));

                thread_pool_init.mode_ = pika::threads::policies::scheduler_mode(
                    pika::threads::policies::do_background_work |
                    pika::threads::policies::reduce_thread_priority |
                    pika::threads::policies::delay_exit);

                std::unique_ptr<pika::threads::thread_pool_base> pool(
                    new pika::threads::detail::scheduled_thread_pool<Scheduler>(
                        std::move(scheduler), thread_pool_init));

                return pool;
            });
    };

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
}

int main(int argc, char* argv[])
{
#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
    {
        using scheduler_type =
            pika::threads::policies::local_priority_queue_scheduler<std::mutex,
                pika::threads::policies::lockfree_lifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }
#endif

    {
        using scheduler_type =
            pika::threads::policies::local_priority_queue_scheduler<std::mutex,
                pika::threads::policies::lockfree_fifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }

#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
    {
        using scheduler_type =
            pika::threads::policies::local_priority_queue_scheduler<std::mutex,
                pika::threads::policies::lockfree_abp_lifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }

    {
        using scheduler_type =
            pika::threads::policies::local_priority_queue_scheduler<std::mutex,
                pika::threads::policies::lockfree_abp_fifo>;
        test_scheduler<scheduler_type>(argc, argv);
    }
#endif

    return pika::util::report_errors();
}
