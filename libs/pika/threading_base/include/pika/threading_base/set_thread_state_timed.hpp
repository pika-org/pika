//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/timing.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/set_thread_state.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#include <atomic>

namespace pika { namespace threads { namespace detail {

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    PIKA_EXPORT thread_id_ref_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        pika::chrono::steady_time_point const& abs_time,
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_schedule_hint schedulehint, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec);

    inline thread_id_ref_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        pika::chrono::steady_time_point const& abs_time,
        thread_id_type const& id, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, abs_time, id,
            thread_schedule_state::pending, thread_restart_state::timeout,
            thread_priority::normal, thread_schedule_hint(), started,
            retry_on_active, ec);
    }

    // Set a timer to set the state of the given \a thread to the given
    // new value after it expired (after the given duration)
    inline thread_id_ref_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        pika::chrono::steady_duration const& rel_time,
        thread_id_type const& thrd, thread_schedule_state newstate,
        thread_restart_state newstate_ex, thread_priority priority,
        thread_schedule_hint schedulehint, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            newstate, newstate_ex, priority, schedulehint, started,
            retry_on_active, ec);
    }

    inline thread_id_ref_type set_thread_state_timed(
        policies::scheduler_base* scheduler,
        pika::chrono::steady_duration const& rel_time,
        thread_id_type const& thrd, std::atomic<bool>* started,
        bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(scheduler, rel_time.from_now(), thrd,
            thread_schedule_state::pending, thread_restart_state::timeout,
            thread_priority::normal, thread_schedule_hint(), started,
            retry_on_active, ec);
    }
}}}    // namespace pika::threads::detail
