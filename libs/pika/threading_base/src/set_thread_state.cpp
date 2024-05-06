//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/functional/bind.hpp>
#include <pika/logging.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/create_work.hpp>
#include <pika/threading_base/register_thread.hpp>
#include <pika/threading_base/set_thread_state.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace pika::threads::detail {
    thread_result_type set_active_state(thread_id_ref_type thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, execution::thread_priority priority,
        thread_state previous_state)
    {
        if (PIKA_UNLIKELY(!thrd))
        {
            PIKA_THROW_EXCEPTION(pika::error::null_thread_id, "threads::detail::set_active_state",
                "null thread id encountered");
            return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
        }

        // make sure that the thread has not been suspended and set active again
        // in the meantime
        thread_state current_state = get_thread_id_data(thrd)->get_state();

        if (current_state.state() == previous_state.state() && current_state != previous_state)
        {
            // NOLINTNEXTLINE(bugprone-branch-clone)
            PIKA_LOG(warn,
                "set_active_state: thread is still active, however it was non-active since the "
                "original set_state request was issued, aborting state change, thread({}), "
                "description({}), new state({})",
                thrd, get_thread_id_data(thrd)->get_description(), get_thread_state_name(newstate));
            return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
        }

        execution::thread_schedule_hint schedulehint{
            static_cast<std::int16_t>(get_thread_id_data(thrd)->get_last_worker_thread_num())};

        // just retry, set_state will create new thread if target is still active
        error_code ec(throwmode::lightweight);    // do not throw
        set_thread_state(thrd.noref(), newstate, newstate_ex, priority, schedulehint, true, ec);

        return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type const& thrd, thread_schedule_state new_state,
        thread_restart_state new_state_ex, execution::thread_priority priority,
        execution::thread_schedule_hint schedulehint, bool retry_on_active, error_code& ec)
    {
        if (PIKA_UNLIKELY(!thrd))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id, "threads::detail::set_thread_state",
                "null thread id encountered");
            return thread_state(thread_schedule_state::unknown, thread_restart_state::unknown);
        }

        // set_state can't be used to force a thread into active state
        if (new_state == thread_schedule_state::active)
        {
            PIKA_THROWS_IF(ec, pika::error::bad_parameter, "threads::detail::set_thread_state",
                "invalid new state: {}", new_state);
            return thread_state(thread_schedule_state::unknown, thread_restart_state::unknown);
        }

        thread_state previous_state;
        std::size_t k = 0;
        do {
            // action depends on the current state
            previous_state = get_thread_id_data(thrd)->get_state();
            thread_schedule_state previous_state_val = previous_state.state();

            // nothing to do here if the state doesn't change
            if (new_state == previous_state_val)
            {
                // NOLINTNEXTLINE(bugprone-branch-clone)
                PIKA_LOG(warn,
                    "set_thread_state: old thread state is the same as new thread state, aborting "
                    "state change, thread({}), description({}), new state({})",
                    thrd, get_thread_id_data(thrd)->get_description(),
                    get_thread_state_name(new_state));

                if (&ec != &throws) ec = make_success_code();

                return thread_state(new_state, previous_state.state_ex());
            }

            // the thread to set the state for is currently running, so we
            // schedule another thread to execute the pending set_state
            switch (previous_state_val)
            {
            case thread_schedule_state::active:
            {
                if (retry_on_active)
                {
                    // schedule a new thread to set the state
                    // NOLINTNEXTLINE(bugprone-branch-clone)
                    PIKA_LOG(warn,
                        "set_thread_state: thread is currently active, scheduling "
                        "new thread, thread({}), description({}), new state({})",
                        thrd, get_thread_id_data(thrd)->get_description(),
                        get_thread_state_name(new_state));

                    thread_init_data data(
                        util::detail::bind(&set_active_state, thread_id_ref_type(thrd), new_state,
                            new_state_ex, priority, previous_state),
                        "set state for active thread", priority, execution::thread_schedule_hint{},
                        execution::thread_stacksize::nostack);

                    create_work(get_thread_id_data(thrd)->get_scheduler_base(), data, ec);

                    if (&ec != &throws) ec = make_success_code();
                }
                else
                {
                    pika::execution::this_thread::detail::yield_k(
                        k, "pika::threads::detail::set_thread_state");
                    ++k;

                    // NOLINTNEXTLINE(bugprone-branch-clone)
                    PIKA_LOG(warn,
                        "set_thread_state: thread is currently active, but not "
                        "scheduling new thread because retry_on_active = false, "
                        "thread({}), description({}), new state({})",
                        thrd, get_thread_id_data(thrd)->get_description(),
                        get_thread_state_name(new_state));

                    continue;
                }

                if (&ec != &throws) ec = make_success_code();

                return previous_state;    // done
            }
            case thread_schedule_state::terminated:
            {
                // NOLINTNEXTLINE(bugprone-branch-clone)
                PIKA_LOG(warn,
                    "set_thread_state: thread is terminated, aborting state "
                    "change, thread({}), description({}), new state({})",
                    thrd, get_thread_id_data(thrd)->get_description(),
                    get_thread_state_name(new_state));

                if (&ec != &throws) ec = make_success_code();

                // If the thread has been terminated while this set_state was
                // pending nothing has to be done anymore.
                return previous_state;
            }
            case thread_schedule_state::pending: [[fallthrough]];
            case thread_schedule_state::pending_boost:
                if (thread_schedule_state::suspended == new_state)
                {
                    // we do not allow explicit resetting of a state to suspended
                    // without the thread being executed.
                    std::string str =
                        fmt::format("set_thread_state: invalid new state, can't demote a pending "
                                    "thread, thread({}), description({}), new state({})",
                            thrd, get_thread_id_data(thrd)->get_description(), new_state);

                    // NOLINTNEXTLINE(bugprone-branch-clone)
                    PIKA_LOG(critical, str);

                    PIKA_THROWS_IF(ec, pika::error::bad_parameter,
                        "threads::detail::set_thread_state", "{}", str);
                    return thread_state(
                        thread_schedule_state::unknown, thread_restart_state::unknown);
                }
                break;
            case thread_schedule_state::suspended: break;    // fine, just set the new state
            case thread_schedule_state::pending_do_not_schedule: [[fallthrough]];
            default:
            {
                PIKA_ASSERT_MSG(false,
                    fmt::format("set_thread_state: previous state was {}",
                        previous_state_val));    // should not happen
                break;
            }
            }

            // If the previous state was pending we are supposed to remove the
            // thread from the queue. But in order to avoid linearly looking
            // through the queue we defer this to the thread function, which
            // at some point will ignore this thread by simply skipping it
            // (if it's not pending anymore).

            // NOLINTNEXTLINE(bugprone-branch-clone)
            PIKA_LOG(info,
                "set_thread_state: thread({}), description({}), new state({}), old state({})", thrd,
                get_thread_id_data(thrd)->get_description(), get_thread_state_name(new_state),
                get_thread_state_name(previous_state_val));

            // So all what we do here is to set the new state.
            if (get_thread_id_data(thrd)->restore_state(new_state, new_state_ex, previous_state))
            {
                break;
            }

            // state has changed since we fetched it from the thread, retry
            // NOLINTNEXTLINE(bugprone-branch-clone)
            PIKA_LOG(warn,
                "set_thread_state: state has been changed since it was fetched, retrying, "
                "thread({}), description({}), new state({}), old state({})",
                thrd, get_thread_id_data(thrd)->get_description(), get_thread_state_name(new_state),
                get_thread_state_name(previous_state_val));
        } while (true);

        thread_schedule_state previous_state_val = previous_state.state();
        if (!(previous_state_val == thread_schedule_state::pending ||
                previous_state_val == thread_schedule_state::pending_boost) &&
            (new_state == thread_schedule_state::pending ||
                new_state == thread_schedule_state::pending_boost))
        {
            // REVIEW: Passing a specific target thread may interfere with the
            // round robin queuing.

            auto* thrd_data = get_thread_id_data(thrd);
            auto* scheduler = thrd_data->get_scheduler_base();
            scheduler->schedule_thread(thrd, schedulehint, false, thrd_data->get_priority());
            // NOTE: Don't care if the hint is a NUMA hint, just want to wake up
            // a thread.
            scheduler->do_some_work(schedulehint.hint);
        }

        if (&ec != &throws) ec = make_success_code();

        return previous_state;
    }
}    // namespace pika::threads::detail
