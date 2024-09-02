//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/assert.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

std::size_t const max_threads =
    (std::min)(std::size_t(4), std::size_t(pika::threads::detail::hardware_concurrency()));

void test_scheduler(int argc, char* argv[], pika::resource::scheduling_policy scheduler)
{
    using ::pika::threads::scheduler_mode;

    pika::init_params init_args;

    init_args.cfg = {"pika.os_threads=" + std::to_string(max_threads)};
    init_args.rp_callback = [scheduler](auto& rp, pika::program_options::variables_map const&) {
        std::size_t pools_added = 0;

        rp.set_default_pool_name("0");
        for (pika::resource::socket const& d : rp.sockets())
        {
            for (pika::resource::core const& c : d.cores())
            {
                for (pika::resource::pu const& p : c.pus())
                {
                    if (pools_added < max_threads)
                    {
                        std::string name = std::to_string(pools_added);
                        rp.create_thread_pool(name, scheduler,
                            scheduler_mode::default_mode | scheduler_mode::enable_elasticity);
                        rp.add_resource(p, name);
                        ++pools_added;
                    }
                }
            }
        }
    };

    pika::start(argc, argv, init_args);

    for (std::size_t pool_num = 0; pool_num < max_threads; ++pool_num)
    {
        auto sched = ex::thread_pool_scheduler{&pika::resource::get_thread_pool(pool_num)};
        tt::sync_wait(ex::schedule(sched) | ex::then([pool_num]() {
            PIKA_TEST_EQ(pika::get_thread_pool_num(), pool_num);
            PIKA_TEST_EQ(pika::get_worker_thread_num(), pool_num);
            PIKA_TEST_EQ(pika::get_local_worker_thread_num(), static_cast<std::size_t>(0));
        }));
    }

    pika::finalize();
    PIKA_TEST_EQ(pika::stop(), 0);
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

    for (auto const scheduler : schedulers) { test_scheduler(argc, argv, scheduler); }

    return 0;
}
