//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test checks for a desirable property in async_rw_mutex: if a previous access is guaranteed
// to have completed, e.g. via sync_wait, the next access is guaranteed to start inline. This makes
// it slightly easier to reason about whether waiting for a sender from async_rw_mutex may yield or
// not.
//
// Note that while we test the property here, we don't guarantee that it won't change in. We simply
// want to preserve the property as long as it's reasonable with the current implementation.

#include <pika/async_rw_mutex.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <cstdlib>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

template <typename M>
void test(M&& m)
{
    ex::thread_pool_scheduler sched{};

    // We first access the mutex in a way such that the wrapper will be released in another task.
    ex::start_detached(m.readwrite() | ex::continues_on(sched) | ex::then([](auto&&) {}));

    // Then we access the mutex again, but block to wait for the result. We discard the result so
    // the wrapper is released immediately.
    {
        [[maybe_unused]] auto wrapper = tt::sync_wait(m.readwrite());
    }

    // Finally, since we blockingly waited for the result above, we expect the below sync_wait to
    // never cause the task yield, or change worker thread. To achieve this, the async_rw_mutex
    // implementation must guarantee that in a situation like this, the wrapper returned by
    // sync_wait holds the last reference to the shared state of that particular access. This would
    // not happen if e.g. the async_rw_mutex_shared_state destructor release the next shared state
    // only once all the continuations have been triggered.
    //
    // We check that neither the thread phase (how many invocations of the tasks, or in other words:
    // did the task yield?) nor worker thread change across the sync_wait. The thread phase is a
    // more reliable check, but is not always available. The worker thread can change if the task
    // yields whenever work stealing is enabled, but is much lower probability.
    auto phase_before = pika::threads::detail::get_self_id_data()->get_thread_phase();
    auto thread_before = pika::get_worker_thread_num();

    {
        [[maybe_unused]] auto wrapper = tt::sync_wait(m.read());
    }

    auto phase_after = pika::threads::detail::get_self_id_data()->get_thread_phase();
    auto thread_after = pika::get_worker_thread_num();

    PIKA_TEST_EQ(phase_before, phase_after);
    PIKA_TEST_EQ(thread_before, thread_after);
}

int pika_main()
{
    pika::scoped_finalize sf{};

    // This whole test fails only with low probability, so repeat it some reasonable number of
    // times. 100 does not guarantee failure in a single run, but hopefully across multiple CI
    // configurations at least one run will fail.
    for (std::size_t iteration = 0; iteration < 100; ++iteration)
    {
        test(ex::async_rw_mutex<int>{42});
        test(ex::async_rw_mutex<void>{});
    }

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
