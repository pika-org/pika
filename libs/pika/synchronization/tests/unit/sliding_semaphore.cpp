//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/synchronization.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> count(0);
int const initial_count = 42;
int const num_tasks = 139;
std::atomic<int> completed_tasks(0);

void worker(pika::sliding_semaphore& sem)
{
    sem.signal(++count);    // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::vector<ex::unique_any_sender<>> senders;
    senders.reserve(num_tasks);

    pika::sliding_semaphore sem(initial_count);

    auto sched = ex::thread_pool_scheduler{};

    for (std::size_t i = 0; i != num_tasks; ++i)
    {
        senders.emplace_back(
            ex::transfer_just(sched, std::ref(sem)) | ex::then(worker) | ex::ensure_started());
    }

    sem.wait(initial_count + num_tasks);

    PIKA_TEST_EQ(count.load(), num_tasks);

    // Since sem.signal(++count) (in worker) is not an atomic operation we wait
    // for the tasks to finish here. The task which signals the count that
    // releases the waiting thread is not necessarily the last one to signal the
    // semaphore. The following can happen:
    //
    //   thread 0             thread 1                thread 2
    //   -------------------  ----------------------- ---------------------
    //   atomic<int> count(0)
    //   semaphore sem(0)
    //   sem.wait(2)
    //        .               new_count = ++count
    //        .                                       new_count = ++count
    //        .                                       sem.signal(new_count)
    //   sem.wait(2) returns
    //   sem destructed
    //                        (sem is a dangling ref)
    //                        sem.signal(new_count)
    //
    tt::sync_wait(ex::when_all_vector(std::move(senders)));

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::init_params init_args;
    init_args.cfg = cfg;
    PIKA_TEST_EQ_MSG(
        pika::init(pika_main, argc, argv, init_args), 0, "pika main exited with non-zero status");

    return 0;
}
