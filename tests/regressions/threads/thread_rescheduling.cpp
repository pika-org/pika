////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <boost/dynamic_bitset.hpp>

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using std::chrono::milliseconds;

using pika::threads::detail::register_thread;

using pika::this_thread::suspend;
using pika::this_thread::yield;
using pika::threads::detail::set_thread_state;
using pika::threads::detail::thread_id_ref_type;
using pika::threads::detail::thread_id_type;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

using sender_vector = std::vector<ex::unique_any_sender<>>;

///////////////////////////////////////////////////////////////////////////////
namespace detail {
    void wait(sender_vector&& senders)
    {
        for (auto& sender : senders)
        {
            tt::sync_wait(std::move(sender));
            yield();
        }
    }
}    // namespace detail

///////////////////////////////////////////////////////////////////////////////
void change_thread_state(thread_id_type thread)
{
    // Do not allow retrying when the thread is active. This can spawn additional tasks that try to
    // set the state and these tasks are not tracked by the sync_waits in pika_main. These calls to
    // set_thread_state can then race against the call to set the state to terminate.
    constexpr bool retry_on_active = false;
    set_thread_state(thread, pika::threads::detail::thread_schedule_state::pending,
        pika::threads::detail::thread_restart_state::signaled,
        pika::execution::thread_priority::normal, retry_on_active);
}

///////////////////////////////////////////////////////////////////////////////
void tree_boot(std::uint64_t count, std::uint64_t grain_size, thread_id_type thread)
{
    PIKA_TEST(grain_size);
    PIKA_TEST(count);

    sender_vector senders;

    std::uint64_t const actors = (count > grain_size) ? grain_size : count;

    std::uint64_t child_count = 0;
    std::uint64_t children = 0;

    if (count > grain_size)
    {
        for (children = grain_size; children != 0; --children)
        {
            child_count = ((count - grain_size) / children);

            if (child_count >= grain_size) break;
        }

        senders.reserve(children + grain_size);
    }
    else { senders.reserve(count); }

    ex::thread_pool_scheduler sched{};
    for (std::uint64_t i = 0; i < children; ++i)
    {
        senders.emplace_back(ex::just(child_count, grain_size, thread) | ex::continues_on(sched) |
            ex::then(tree_boot) | ex::ensure_started());
    }

    for (std::uint64_t i = 0; i < actors; ++i)
    {
        senders.emplace_back(ex::just(thread) | ex::continues_on(sched) |
            ex::then(change_thread_state) | ex::ensure_started());
    }

    detail::wait(std::move(senders));
}

bool woken = false;

///////////////////////////////////////////////////////////////////////////////
void test_dummy_thread()
{
    while (true)
    {
        pika::threads::detail::thread_restart_state statex =
            suspend(pika::threads::detail::thread_schedule_state::suspended);

        if (statex == pika::threads::detail::thread_restart_state::terminate)
        {
            woken = true;
            return;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    std::uint64_t const tasks = vm["tasks"].as<std::uint64_t>();
    std::uint64_t const grain_size = vm["grain-size"].as<std::uint64_t>();

    {
        pika::threads::detail::thread_init_data data(
            pika::threads::detail::make_thread_function_nullary(test_dummy_thread),
            "test_dummy_thread");
        thread_id_ref_type thread_id = register_thread(data);
        PIKA_TEST_NEQ(thread_id, pika::threads::detail::invalid_thread_id);

        ex::thread_pool_scheduler sched{};
        // Flood the queues with set_thread_state operations before the rescheduling attempt.
        auto before = ex::just(tasks, grain_size, thread_id.noref()) | ex::continues_on(sched) |
            ex::then(tree_boot) | ex::ensure_started();

        set_thread_state(thread_id.noref(), pika::threads::detail::thread_schedule_state::pending,
            pika::threads::detail::thread_restart_state::signaled);

        // Flood the queues with set_thread_state operations after the rescheduling attempt.
        auto after = ex::just(tasks, grain_size, thread_id.noref()) | ex::continues_on(sched) |
            ex::then(tree_boot) | ex::ensure_started();

        tt::sync_wait(std::move(before));
        tt::sync_wait(std::move(after));

        // set_thread_state will not change the restart state if the task is already in a pending
        // state. Even though we wait for the tasks above to finish, the target task may not be
        // active or suspended yet. Since we want to guarantee that the task terminates, we may have
        // to retry multiple times until the restart state actually becomes terminate.
        pika::util::yield_while([&] {
            auto prev_state = set_thread_state(thread_id.noref(),
                pika::threads::detail::thread_schedule_state::pending,
                pika::threads::detail::thread_restart_state::terminate);
            return prev_state.state() == pika::threads::detail::thread_schedule_state::pending;
        });
    }

    pika::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("tasks", value<std::uint64_t>()->default_value(64),
         "number of tasks to invoke before and after the rescheduling")
        ("grain-size", value<std::uint64_t>()->default_value(4), "grain size of the future tree");
    // clang-format on

    // Initialize and run pika
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    PIKA_TEST_EQ(0, pika::init(pika_main, argc, argv, init_args));

    PIKA_TEST(woken);

    return 0;
}
