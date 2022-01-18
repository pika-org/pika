//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test creates a random number of thread pools at startup
// and assigns a random number of threads to each (not exceeding the
// number of threads available in total). Tasks are created and assigned
// to random pools, with continuations assigned to another random pool
// using an executor per pool.
// The test is intended to stress test the scheduler and ensure that
// cross pool injection of tasks does not cause segfaults or other
// problems such as lockups.

#include <pika/local/execution.hpp>
#include <pika/local/functional.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threadmanager.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

std::size_t num_pools = 0;

// dummy function we will call using async
void dummy_task(std::size_t n)
{
    // no other work can take place on this thread whilst it sleeps
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
    //
    for (std::size_t i(0); i < n; ++i)
    {
    }
}

inline std::size_t st_rand(std::size_t a, std::size_t b)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<std::size_t> dist(a, b);
    //
    return std::size_t(dist(gen));
}

int pika_main()
{
    PIKA_TEST_EQ(std::size_t(0), pika::resource::get_pool_index("default"));
    PIKA_TEST_EQ(std::size_t(0), pika::resource::get_pool_index("pool-0"));

    // print partition characteristics
    pika::threads::get_thread_manager().print_pools(std::cout);

    auto const sched = pika::threads::get_self_id_data()->get_scheduler_base();
    if (std::string("core-shared_priority_queue_scheduler") ==
        sched->get_description())
    {
        sched->add_remove_scheduler_mode(
            // add these flags
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::enable_stealing |
                pika::threads::policies::enable_stealing_numa |
                pika::threads::policies::assign_work_thread_parent |
                pika::threads::policies::steal_after_local),
            // remove these flags
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::assign_work_round_robin |
                pika::threads::policies::steal_high_priority_first));
        sched->update_scheduler_mode(
            pika::threads::policies::enable_stealing, false);
        sched->update_scheduler_mode(
            pika::threads::policies::enable_stealing_numa, false);
    }

    // setup executors for different task priorities on the pools
    std::vector<pika::execution::parallel_executor> HP_executors;
    std::vector<pika::execution::parallel_executor> NP_executors;
    for (std::size_t i = 0; i < num_pools; ++i)
    {
        std::string pool_name = "pool-" + std::to_string(i);
        HP_executors.emplace_back(&pika::resource::get_thread_pool(pool_name),
            pika::threads::thread_priority::high);
        NP_executors.emplace_back(&pika::resource::get_thread_pool(pool_name),
            pika::threads::thread_priority::default_);
    }

    // randomly create tasks that run on a random pool
    // attach continuations to them that run on different
    // random pools
    int const loops = 1000;
    //
    std::cout << "1: Starting HP " << loops << std::endl;
    std::atomic<int> counter(loops);
    for (int i = 0; i < loops; ++i)
    {
        // high priority
        std::size_t random_pool_1 = st_rand(0, num_pools - 1);
        std::size_t random_pool_2 = st_rand(0, num_pools - 1);
        auto& exec_1 = HP_executors[random_pool_1];
        auto& exec_2 = HP_executors[random_pool_2];
        auto f1 = pika::async(exec_1, &dummy_task, 0);
        auto f2 = f1.then(exec_2, [=, &counter](pika::future<void>&&) {
            dummy_task(0);
            --counter;
        });
    }
    do
    {
        pika::this_thread::yield();
    } while (counter > 0);

    std::cout << "2: Starting NP " << loops << std::endl;
    counter = loops;
    for (int i = 0; i < loops; ++i)
    {
        // normal priority
        std::size_t random_pool_1 = st_rand(0, num_pools - 1);
        std::size_t random_pool_2 = st_rand(0, num_pools - 1);
        auto& exec_3 = NP_executors[random_pool_1];
        auto& exec_4 = NP_executors[random_pool_2];
        auto f3 = pika::async(exec_3, &dummy_task, 0);
        auto f4 = f3.then(exec_4, [=, &counter](pika::future<void>&&) {
            dummy_task(0);
            --counter;
        });
    }
    do
    {
        pika::this_thread::yield();
    } while (counter > 0);

    std::cout << "3: Starting HP->NP " << loops << std::endl;
    counter = loops;
    for (int i = 0; i < loops; ++i)
    {
        // mixed priority, HP->NP
        std::size_t random_pool_1 = st_rand(0, num_pools - 1);
        std::size_t random_pool_2 = st_rand(0, num_pools - 1);
        auto& exec_5 = HP_executors[random_pool_1];
        auto& exec_6 = NP_executors[random_pool_2];
        auto f5 = pika::async(exec_5, &dummy_task, 0);
        auto f6 = f5.then(exec_6, [=, &counter](pika::future<void>&&) {
            dummy_task(0);
            --counter;
        });
    }
    do
    {
        pika::this_thread::yield();
    } while (counter > 0);

    std::cout << "4: Starting NP->HP " << loops << std::endl;
    counter = loops;
    for (int i = 0; i < loops; ++i)
    {
        // mixed priority, NP->HP
        std::size_t random_pool_1 = st_rand(0, num_pools - 1);
        std::size_t random_pool_2 = st_rand(0, num_pools - 1);
        auto& exec_7 = NP_executors[random_pool_1];
        auto& exec_8 = HP_executors[random_pool_2];
        auto f7 = pika::async(exec_7, &dummy_task, 0);
        auto f8 = f7.then(exec_8, [=, &counter](pika::future<void>&&) {
            dummy_task(0);
            --counter;
        });
    }
    do
    {
        pika::this_thread::yield();
    } while (counter > 0);

    std::cout << "5: Starting suspending " << loops << std::endl;
    counter = loops;
    for (int i = 0; i < loops; ++i)
    {
        // tasks that depend on each other and need to suspend
        std::size_t random_pool_1 = st_rand(0, num_pools - 1);
        std::size_t random_pool_2 = st_rand(0, num_pools - 1);
        auto& exec_7 = NP_executors[random_pool_1];
        auto& exec_8 = HP_executors[random_pool_2];
        // random delay up to 5 milliseconds
        std::size_t delay = st_rand(0, 5);
        auto f7 = pika::async(exec_7, &dummy_task, delay);
        auto f8 = pika::async(exec_8, [f7(std::move(f7)), &counter]() mutable {
            // if f7 is not ready then f8 will suspend itself on get
            f7.get();
            dummy_task(0);
            --counter;
        });
    }
    do
    {
        pika::this_thread::yield();
    } while (counter > 0);

    return pika::local::finalize();
}

void init_resource_partitioner_handler(pika::resource::partitioner& rp,
    pika::program_options::variables_map const&,
    pika::resource::scheduling_policy policy)
{
    num_pools = 0;

    // before adding pools - set the default pool name to "pool-0"
    rp.set_default_pool_name("pool-0");

    auto seed = std::time(nullptr);
    std::srand(seed);
    std::cout << "Random seed " << seed << std::endl;

    // create N pools
    std::size_t max_threads = rp.get_number_requested_threads();
    std::string pool_name;
    std::size_t threads_remaining = max_threads;
    std::size_t threads_in_pool = 0;
    // create pools randomly and add a random number of PUs to each pool
    for (pika::resource::numa_domain const& d : rp.numa_domains())
    {
        for (pika::resource::core const& c : d.cores())
        {
            for (pika::resource::pu const& p : c.pus())
            {
                if (threads_in_pool == 0)
                {
                    // pick a random number of threads less than the max
                    threads_in_pool = 1 +
                        st_rand(
                            0, ((std::max)(std::size_t(1), max_threads / 2)));
                    pool_name = "pool-" + std::to_string(num_pools);
                    rp.create_thread_pool(pool_name, policy);
                    num_pools++;
                }
                std::cout << "Added pu " << p.id() << " to " << pool_name
                          << "\n";
                rp.add_resource(p, pool_name);
                threads_in_pool--;
                if (threads_remaining-- == 0)
                {
                    std::cerr << "This should not happen!" << std::endl;
                }
            }
        }
    }
}

void test_scheduler(
    int argc, char* argv[], pika::resource::scheduling_policy scheduler)
{
    pika::local::init_params init_args;
    init_args.rp_callback =
        pika::bind_back(init_resource_partitioner_handler, scheduler);
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
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
        // The shared_priority scheduler sometimes hangs in this test.
        //pika::resource::scheduling_policy::shared_priority,
    };

    for (auto const scheduler : schedulers)
    {
        test_scheduler(argc, argv, scheduler);
    }

    return pika::util::report_errors();
}
