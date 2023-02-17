//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/chrono.hpp>
#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>
#include <pika/thread_pool_util/thread_pool_suspension_helpers.hpp>
#include <pika/threading_base/thread_helpers.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

void test_scheduler(int argc, char* argv[], pika::resource::scheduling_policy scheduler)
{
    pika::init_params init_args;

    init_args.cfg = {"pika.os_threads=" +
        std::to_string(((std::min)(
            std::size_t(4), std::size_t(pika::threads::detail::hardware_concurrency()))))};
    init_args.rp_callback = [scheduler](auto& rp, pika::program_options::variables_map const&) {
        rp.create_thread_pool("default", scheduler);
    };

    pika::start(nullptr, argc, argv, init_args);

    pika::threads::detail::thread_pool_base& default_pool =
        pika::resource::get_thread_pool("default");
    std::size_t const default_pool_threads = pika::resource::get_num_threads("default");

    pika::chrono::detail::high_resolution_timer t;

    while (t.elapsed() < 2)
    {
        for (std::size_t i = 0; i < default_pool_threads * 10000; ++i)
        {
            pika::apply([]() {});
        }

        bool suspended = false;
        pika::threads::detail::suspend_pool_cb(default_pool, [&suspended]() { suspended = true; });

        // NOLINTNEXTLINE(bugprone-infinite-loop)
        while (!suspended)
        {
            std::this_thread::yield();
        }

        bool resumed = false;
        pika::threads::detail::resume_pool_cb(default_pool, [&resumed]() { resumed = true; });

        // NOLINTNEXTLINE(bugprone-infinite-loop)
        while (!resumed)
        {
            std::this_thread::yield();
        }
    }

    pika::apply([]() { pika::finalize(); });

    PIKA_TEST_EQ(pika::stop(), 0);
}

int main(int argc, char* argv[])
{
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

    return 0;
}
