//  Copyright (c) 2020 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test checks that no thread has thread_stacksize::current as its actual
// stacksize. thread_stacksize::current can be used as input when creating a
// thread, but it should always be converted to something between
// thread_stacksize::minimal and thread_stacksize::maximal when a thread has been
// created.

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

void test(pika::execution::thread_stacksize stacksize)
{
    auto sched = ex::with_stacksize(ex::thread_pool_scheduler{}, stacksize);
    auto sched_current =
        ex::with_stacksize(ex::thread_pool_scheduler{}, pika::execution::thread_stacksize::current);

    tt::sync_wait(ex::schedule(sched) | ex::then([&sched_current, stacksize]() {
        // This thread should have the stack size stacksize; it has been
        // explicitly set in the scheduler.
        pika::execution::thread_stacksize self_stacksize =
            pika::threads::detail::get_self_stacksize_enum();
        PIKA_TEST_EQ(self_stacksize, stacksize);
        PIKA_TEST_NEQ(self_stacksize, pika::execution::thread_stacksize::current);

        tt::sync_wait(ex::schedule(sched_current) | ex::then([stacksize]() {
            // This thread should also have the stack size stacksize; it has
            // been inherited size from the parent thread.
            pika::execution::thread_stacksize self_stacksize =
                pika::threads::detail::get_self_stacksize_enum();
            PIKA_TEST_EQ(self_stacksize, stacksize);
            PIKA_TEST_NEQ(self_stacksize, pika::execution::thread_stacksize::current);
        }));
    }));
}

int pika_main()
{
    for (pika::execution::thread_stacksize stacksize = pika::execution::thread_stacksize::minimal;
         stacksize < pika::execution::thread_stacksize::maximal;
         stacksize = static_cast<pika::execution::thread_stacksize>(
             static_cast<std::size_t>(stacksize) + 1))
    {
        test(stacksize);
    }

    return pika::finalize();
}

int main(int argc, char** argv)
{
    std::vector<std::string> schedulers = {"local", "local-priority-fifo", "local-priority-lifo",
        "static", "static-priority", "abp-priority-fifo", "abp-priority-lifo", "shared-priority"};
    for (auto const& scheduler : schedulers)
    {
        pika::init_params iparams;
        iparams.cfg = {"--pika:queuing=" + std::string(scheduler)};
        std::cout << iparams.cfg[0] << std::endl;
        pika::init(pika_main, argc, argv, iparams);
    }

    return 0;
}
