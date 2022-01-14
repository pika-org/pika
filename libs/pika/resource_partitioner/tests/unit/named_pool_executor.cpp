//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource partitioner
// pool and executor

#include <pika/assert.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

std::size_t const max_threads = (std::min)(
    std::size_t(4), std::size_t(pika::threads::hardware_concurrency()));

// dummy function we will call using async
void dummy_task(std::size_t n, std::string const& text)
{
    for (std::size_t i(0); i < n; ++i)
    {
        std::cout << text << " iteration " << i << "\n";
    }
}

int pika_main()
{
    PIKA_TEST_EQ(std::size_t(max_threads), pika::resource::get_num_threads());
    PIKA_TEST_EQ(
        std::size_t(max_threads), pika::resource::get_num_thread_pools());
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
    pika::execution::parallel_executor exec_default;
    PIKA_UNUSED(exec_default);

    // setup executors for different task priorities on the pools
    // segfaults or exceptions in any of the following will cause
    // the test to fail
    pika::execution::parallel_executor exec_0_hp(
        &pika::resource::get_thread_pool("default"),
        pika::threads::thread_priority::high);

    pika::execution::parallel_executor exec_0(
        &pika::resource::get_thread_pool("default"),
        pika::threads::thread_priority::default_);

    std::vector<pika::future<void>> lotsa_futures;

    // use executors to schedule work on pools
    lotsa_futures.push_back(
        pika::async(exec_0_hp, &dummy_task, 3, "HP default"));

    lotsa_futures.push_back(
        pika::async(exec_0, &dummy_task, 3, "Normal default"));

    std::vector<pika::execution::parallel_executor> execs;
    std::vector<pika::execution::parallel_executor> execs_hp;
    //
    for (std::size_t i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        execs.push_back(pika::execution::parallel_executor(
            &pika::resource::get_thread_pool(pool_name),
            pika::threads::thread_priority::default_));
        execs_hp.push_back(pika::execution::parallel_executor(
            &pika::resource::get_thread_pool(pool_name),
            pika::threads::thread_priority::high));
    }

    for (std::size_t i = 0; i < max_threads; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        lotsa_futures.push_back(
            pika::async(execs[i], &dummy_task, 3, pool_name + " normal"));
        lotsa_futures.push_back(
            pika::async(execs_hp[i], &dummy_task, 3, pool_name + " HP"));
    }

    // check that the default executor still works
    pika::execution::parallel_executor large_stack_executor(
        pika::threads::thread_stacksize::large);

    lotsa_futures.push_back(pika::async(
        large_stack_executor, &dummy_task, 3, "true default + large stack"));

    // just wait until everything is done
    pika::when_all(lotsa_futures).get();

    return pika::local::finalize();
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
        rp.create_thread_pool(
            pool_name, pika::resource::scheduling_policy::local_priority_fifo);
    }

    // add one PU to each pool
    std::size_t thread_count = 0;
    for (pika::resource::numa_domain const& d : rp.numa_domains())
    {
        for (pika::resource::core const& c : d.cores())
        {
            for (pika::resource::pu const& p : c.pus())
            {
                if (thread_count < max_threads)
                {
                    std::string pool_name =
                        "pool-" + std::to_string(thread_count);
                    std::cout << "Added pu " << thread_count << " to "
                              << pool_name << "\n";
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

    pika::local::init_params init_args;
    init_args.cfg = {"pika.os_threads=" + std::to_string(max_threads)};
    // Set the callback to init the thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    // now run the test
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
    return pika::util::report_errors();
}
