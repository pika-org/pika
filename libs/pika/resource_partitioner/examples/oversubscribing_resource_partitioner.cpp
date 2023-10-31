//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
# include <pika/execution.hpp>
# include <pika/init.hpp>
# include <pika/modules/resource_partitioner.hpp>
# include <pika/modules/schedulers.hpp>
# include <pika/runtime.hpp>
# include <pika/thread.hpp>
//
# include <cmath>
# include <cstddef>
# include <cstdint>
# include <iostream>
# include <memory>
# include <set>
# include <string>
# include <utility>
# include <vector>
//
# include "system_characteristics.hpp"

// NB
// this test needs to be updated as it no longer does what it is supposed to do

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

namespace resource::pools {
    enum ids
    {
        DEFAULT = 0,
        MPI = 1,
        GPU = 2,
        MATRIX = 3,
    };
}    // namespace resource::pools

static bool use_pools = false;
static int pool_threads = 1;
static std::string const pool_name = "mpi";

// this is our custom scheduler type
using high_priority_sched = pika::threads::detail::shared_priority_queue_scheduler<>;
using namespace pika::threads;
using pika::threads::scheduler_mode;

// dummy function we will call using then
void do_stuff(std::size_t n, bool printout)
{
    if (printout) std::cout << "[do stuff] " << n << "\n";
    for (std::size_t i(0); i < n; ++i)
    {
        double f = std::sin(2 * M_PI * static_cast<double>(i) / static_cast<double>(n));
        if (printout) std::cout << "sin(" << i << ") = " << f << ", ";
    }
    if (printout) std::cout << "\n";
}

// this is called on a pika thread after the runtime starts up
int pika_main(/*pika::program_options::variables_map& vm*/)
{
    std::size_t num_threads = pika::get_num_worker_threads();
    std::cout << "pika using threads = " << num_threads << std::endl;

    if (num_threads == 1)
    {
        PIKA_THROW_EXCEPTION(pika::error::commandline_option_error, "pika_main",
            "the oversubscribing_resource_partitioner example requires at least 2 worker threads "
            "(1 given)");
    }

    std::size_t loop_count = num_threads * 1;
    std::size_t async_count = num_threads * 1;

    // create a scheduler with high priority for important tasks
    auto high_priority_scheduler =
        ex::with_priority(ex::thread_pool_scheduler{pika::this_thread::get_pool()},
            pika::execution::thread_priority::high_recursive);
    auto normal_priority_scheduler = ex::thread_pool_scheduler{pika::this_thread::get_pool()};

    ex::thread_pool_scheduler pool_scheduler;
    // create a scheduler on the mpi pool
    if (use_pools)
    {
        // get schedulers
        pool_scheduler = ex::thread_pool_scheduler{&pika::resource::get_thread_pool(pool_name)};
        std::cout << "\n[pika_main] got mpi scheduler " << std::endl;
    }
    else { pool_scheduler = high_priority_scheduler; }

    // print partition characteristics
    std::cout << "\n\n[pika_main] print resource_partitioner characteristics : "
              << "\n";
    pika::resource::get_partitioner().print_init_pool_data(std::cout);

    // print partition characteristics
    std::cout << "\n\n[pika_main] print thread-manager pools : "
              << "\n";
    pika::threads::get_thread_manager().print_pools(std::cout);

    // print system characteristics
    print_system_characteristics();

    // use scheduler to schedule work on custom pool
    auto sender = ex::schedule(pool_scheduler) |
        ex::then(pika::util::detail::bind_front(do_stuff, 5, true)) | ex::transfer(pool_scheduler) |
        ex::then([]() { do_stuff(5, true); }) | ex::transfer(pool_scheduler) |
        ex::then([pool_scheduler, high_priority_scheduler, async_count]() mutable {
            ex::unique_any_sender<> sender1, sender2;
            for (std::size_t i = 0; i < async_count; i++)
            {
                if (i % 2 == 0)
                {
                    sender1 = ex::schedule(pool_scheduler) |
                        ex::then(pika::util::detail::bind_front(&do_stuff, async_count, false));
                }
                else
                {
                    sender2 = ex::schedule(high_priority_scheduler) |
                        ex::then(pika::util::detail::bind_front(&do_stuff, async_count, false));
                }
            }

            // the last senders we made are stored in here
            tt::sync_wait(std::move(sender1));
            tt::sync_wait(std::move(sender2));
        });

    tt::sync_wait(std::move(sender));

    pika::mutex m;
    std::set<std::thread::id> thread_set;
    std::vector<ex::unique_any_sender<>> senders;

    // launch tasks on custom pool with high priority
    for (std::size_t i = 0; i < loop_count; ++i)
    {
        senders.push_back(ex::schedule(high_priority_scheduler) | ex::then([&, i]() {
            std::lock_guard<pika::mutex> lock(m);
            if (thread_set.insert(std::this_thread::get_id()).second)
            {
                std::cout << std::hex << pika::this_thread::get_id() << " " << std::hex
                          << std::this_thread::get_id() << " high priority scheduler i " << std::dec
                          << i << std::endl;
            }
        }));
    }
    tt::sync_wait(ex::when_all_vector(std::move(senders)));
    std::cout << "thread set contains " << std::dec << thread_set.size() << std::endl;
    senders.clear();
    thread_set.clear();

    // launch tasks on custom pool with normal priority
    for (std::size_t i = 0; i < loop_count; ++i)
    {
        senders.push_back(ex::schedule(normal_priority_scheduler) | ex::then([&, i]() {
            std::lock_guard<pika::mutex> lock(m);
            if (thread_set.insert(std::this_thread::get_id()).second)
            {
                std::cout << std::hex << pika::this_thread::get_id() << " " << std::hex
                          << std::this_thread::get_id() << " normal priority scheduler i "
                          << std::dec << i << std::endl;
            }
        }));
    }
    tt::sync_wait(ex::when_all_vector(std::move(senders)));
    std::cout << "thread set contains " << std::dec << thread_set.size() << std::endl;
    senders.clear();
    thread_set.clear();

    // test a parallel algorithm on pool_scheduler
    for (std::size_t i = 0; i < loop_count; ++i)
    {
        senders.push_back(ex::schedule(pool_scheduler) | ex::then([&, i]() {
            std::lock_guard<pika::mutex> lock(m);
            if (thread_set.insert(std::this_thread::get_id()).second)
            {
                std::cout << std::hex << pika::this_thread::get_id() << " " << std::hex
                          << std::this_thread::get_id() << " pool scheduler i " << std::dec << i
                          << std::endl;
            }
        }));
    }
    tt::sync_wait(ex::when_all_vector(std::move(senders)));
    std::cout << "thread set contains " << std::dec << thread_set.size() << std::endl;
    senders.clear();
    thread_set.clear();

    return pika::finalize();
}

// -------------------------------------------------------------------------
void init_resource_partitioner_handler(
    pika::resource::partitioner& rp, pika::program_options::variables_map const& vm)
{
    use_pools = vm.count("use-pools") != 0;
    pool_threads = vm["pool-threads"].as<int>();

    std::cout << "[pika_main] starting ..."
              << "use_pools " << use_pools << " "
              << "pool-threads " << pool_threads << "\n";

    if (pool_threads > 0)
    {
        // we use unspecified as the scheduler type and it will be set according to
        // the --pika:queuing=xxx option or default.
        auto deft = ::pika::threads::scheduler_mode::default_mode;
        rp.create_thread_pool(pool_name, pika::resource::scheduling_policy::shared_priority, deft);
        // add N pus to network pool
        int count = 0;
        for (pika::resource::numa_domain const& d : rp.numa_domains())
        {
            for (pika::resource::core const& c : d.cores())
            {
                for (pika::resource::pu const& p : c.pus())
                {
                    if (count < pool_threads)
                    {
                        std::cout << "Added pu " << count++ << " to pool \"" << pool_name << "\"\n";
                        rp.add_resource(p, pool_name);
                    }
                }
            }
        }

        rp.create_thread_pool("default", pika::resource::scheduling_policy::unspecified, deft);
    }
}

// the normal int main function that is called at startup and runs on an OS
// thread the user must call pika::init to start the pika runtime which
// will execute pika_main on a pika thread
int main(int argc, char* argv[])
{
    // clang-format off
    pika::program_options::options_description desc_cmdline("Test options");
    desc_cmdline.add_options()
        ("use-pools,u", "Enable advanced pika thread pools and schedulers")
        ("use-scheduler,s", "Enable custom priority scheduler")
        ("pool-threads,m", pika::program_options::value<int>()->default_value(1),
            "Number of threads to assign to custom pool");
    // clang-format on

    pika::init_params iparams;

    iparams.desc_cmdline = desc_cmdline;
    iparams.rp_mode = pika::resource::mode_allow_oversubscription;
    iparams.rp_callback = init_resource_partitioner_handler;

    return pika::init(pika_main, argc, argv, iparams);
}
#endif
