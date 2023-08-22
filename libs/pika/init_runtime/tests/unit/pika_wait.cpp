//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This test checks that the runtime takes into account suspended threads before
// initiating full shutdown.

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/runtime/thread_pool_helpers.hpp>
#include <pika/testing.hpp>

#include <cstddef>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

using pika::threads::detail::thread_schedule_state;

void test_wait()
{
    for (std::size_t i = 0; i < 1000; ++i)
    {
        ex::execute(ex::thread_pool_scheduler{}, [] {});
    }

    pika::wait();

    if (pika::threads::detail::get_self_ptr())
    {
        PIKA_TEST_EQ(pika::threads::get_thread_count(thread_schedule_state::active), 1);
    }
    else { PIKA_TEST_EQ(pika::threads::get_thread_count(thread_schedule_state::active), 0); }

    PIKA_TEST_EQ(pika::threads::get_thread_count(thread_schedule_state::pending), 0);
    PIKA_TEST_EQ(pika::threads::get_thread_count(thread_schedule_state::suspended), 0);
    PIKA_TEST_EQ(pika::threads::get_thread_count(thread_schedule_state::staged), 0);
}

int main(int argc, char** argv)
{
    pika::start(nullptr, argc, argv);

    // Test outside the runtime
    test_wait();

    // Test in a pika thread
    tt::sync_wait(ex::schedule(ex::thread_pool_scheduler()) | ex::then(test_wait));

    // Test in a stackless pika thread
    tt::sync_wait(ex::schedule(ex::with_stacksize(
                      ex::thread_pool_scheduler(), pika::execution::thread_stacksize::nostack)) |
        ex::then(test_wait));

    pika::finalize();
    pika::stop();

    return 0;
}
