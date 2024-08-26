//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource partitioner
// pool and scheduler

#include <pika/assert.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

std::size_t const max_threads =
    (std::min)(std::size_t(4), std::size_t(pika::threads::detail::hardware_concurrency()));

// dummy function we will call using async
void dummy_task(std::size_t n, std::string const& text)
{
    for (std::size_t i(0); i < n; ++i) { std::cout << text << " iteration " << i << "\n"; }
}

int pika_main()
{
    PIKA_TEST_EQ(std::size_t(max_threads), pika::resource::get_num_threads());
    PIKA_TEST_EQ(std::size_t(max_threads), pika::resource::get_num_thread_pools());
    PIKA_TEST_EQ(std::size_t(0), pika::resource::get_pool_index("default"));
    PIKA_TEST_EQ(std::size_t(0), pika::resource::get_pool_index("pool-0"));
    PIKA_TEST(pika::resource::pool_exists("default"));
    PIKA_TEST(pika::resource::pool_exists("pool-0"));
    PIKA_TEST(!pika::resource::pool_exists("nonexistent"));
    for (std::size_t pool_index = 0; pool_index < max_threads; ++pool_index)
    {
        PIKA_TEST(pika::resource::pool_exists(pool_index));
    }
    PIKA_TEST(!pika::resource::pool_exists(max_threads));

    for (std::size_t i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        PIKA_TEST_EQ(pool_name, pika::resource::get_pool_name(i));
        PIKA_TEST_EQ(std::size_t(1), pika::resource::get_num_threads(i));
    }

    // Make sure default construction works
    [[maybe_unused]] ex::thread_pool_scheduler sched_default;

    // setup schedulers for different task priorities on the pools
    // segfaults or exceptions in any of the following will cause
    // the test to fail
    auto sched_0_hp =
        ex::with_priority(ex::thread_pool_scheduler{&pika::resource::get_thread_pool("default")},
            pika::execution::thread_priority::high);

    auto sched_0 =
        ex::with_priority(ex::thread_pool_scheduler{&pika::resource::get_thread_pool("default")},
            pika::execution::thread_priority::default_);

    std::vector<ex::unique_any_sender<>> lotsa_senders;

    // use schedulers to schedule work on pools
    lotsa_senders.push_back(ex::transfer_just(sched_0_hp, 3, "HP default") | ex::then(dummy_task));
    lotsa_senders.push_back(ex::transfer_just(sched_0, 3, "Normal default") | ex::then(dummy_task));

    std::vector<ex::thread_pool_scheduler> scheds;
    std::vector<ex::thread_pool_scheduler> scheds_hp;
    //
    for (std::size_t i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        scheds.push_back(ex::with_priority(
            ex::thread_pool_scheduler{&pika::resource::get_thread_pool(pool_name)},
            pika::execution::thread_priority::default_));
        scheds_hp.push_back(ex::with_priority(
            ex::thread_pool_scheduler{&pika::resource::get_thread_pool(pool_name)},
            pika::execution::thread_priority::high));
    }

    for (std::size_t i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        lotsa_senders.push_back(
            ex::transfer_just(scheds[i], 3, pool_name + "normal") | ex::then(dummy_task));
        lotsa_senders.push_back(
            ex::transfer_just(scheds_hp[i], 3, pool_name + " HP") | ex::then(dummy_task));
    }

    // check that the default scheduler still works
    auto large_stack_scheduler =
        ex::with_stacksize(ex::thread_pool_scheduler{}, pika::execution::thread_stacksize::large);

    lotsa_senders.push_back(
        ex::transfer_just(large_stack_scheduler, 3, "true default + large stack") |
        ex::then(dummy_task));

    // just wait until everything is done
    tt::sync_wait(ex::when_all_vector(std::move(lotsa_senders)));

    pika::finalize();
    return EXIT_SUCCESS;
}

void init_resource_partitioner_handler(
    pika::resource::partitioner& rp, pika::program_options::variables_map const&)
{
    // before adding pools - set the default pool name to "pool-0"
    rp.set_default_pool_name("pool-0");

    // create N pools
    for (std::size_t i = 0; i < max_threads; i++)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        rp.create_thread_pool(pool_name, pika::resource::scheduling_policy::local_priority_fifo);
    }

    // add one PU to each pool
    std::size_t thread_count = 0;
    for (pika::resource::socket const& d : rp.sockets())
    {
        for (pika::resource::core const& c : d.cores())
        {
            for (pika::resource::pu const& p : c.pus())
            {
                if (thread_count < max_threads)
                {
                    std::string pool_name = "pool-" + std::to_string(thread_count);
                    std::cout << "Added pu " << thread_count << " to " << pool_name << "\n";
                    rp.add_resource(p, pool_name);
                    thread_count++;
                }
            }
        }
    }
}

// this test must be run with at least 2 threads
int main(int argc, char* argv[])
{
    PIKA_ASSERT(max_threads >= 2);

    pika::init_params init_args;
    init_args.cfg = {"pika.os_threads=" + std::to_string(max_threads)};
    // Set the callback to init the thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    // now run the test
    PIKA_TEST_EQ(pika::init(pika_main, argc, argv, init_args), 0);
    return 0;
}
