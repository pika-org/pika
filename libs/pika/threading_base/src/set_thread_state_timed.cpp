//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/functional/bind.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/create_thread.hpp>
#include <pika/threading_base/set_thread_state_timed.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <system_error>
#include <utility>

namespace pika::threads::detail {
    /// This thread function is used by the at_timer thread below to trigger
    /// the required action.
    thread_result_type wake_timer_thread(thread_id_ref_type const& thrd,
        thread_schedule_state /*newstate*/, thread_restart_state /*newstate_ex*/,
        execution::thread_priority /*priority*/, thread_id_type timer_id,
        std::shared_ptr<std::atomic<bool>> const& triggered, bool retry_on_active,
        thread_restart_state my_statex)
    {
        if (PIKA_UNLIKELY(!thrd))
        {
            PIKA_THROW_EXCEPTION(pika::error::null_thread_id, "threads::detail::wake_timer_thread",
                "null thread id encountered (id)");
            return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
        }

        if (PIKA_UNLIKELY(!timer_id))
        {
            PIKA_THROW_EXCEPTION(pika::error::null_thread_id, "threads::detail::wake_timer_thread",
                "null thread id encountered (timer_id)");
            return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
        }

        PIKA_ASSERT(
            my_statex == thread_restart_state::abort || my_statex == thread_restart_state::timeout);

        if (!triggered->load())
        {
            error_code ec(throwmode::lightweight);    // do not throw
            set_thread_state(timer_id, thread_schedule_state::pending, my_statex,
                execution::thread_priority::boost, execution::thread_schedule_hint(),
                retry_on_active, ec);
        }

        return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
    }

    /// This thread function initiates the required set_state action (on
    /// behalf of one of the threads#detail#set_thread_state functions).
    thread_result_type at_timer(scheduler_base* scheduler,
        std::chrono::steady_clock::time_point& /*abs_time*/, thread_id_ref_type const& thrd,
        thread_schedule_state newstate, thread_restart_state newstate_ex,
        execution::thread_priority priority, std::atomic<bool>* /*started*/, bool retry_on_active)
    {
        if (PIKA_UNLIKELY(!thrd))
        {
            PIKA_THROW_EXCEPTION(pika::error::null_thread_id, "threads::detail::at_timer",
                "null thread id encountered");
            return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
        }

        // create a new thread in suspended state, which will execute the
        // requested set_state when timer fires and will re-awaken this thread,
        // allowing the deadline_timer to go out of scope gracefully
        thread_id_ref_type self_id = get_self_id();    // keep alive

        std::shared_ptr<std::atomic<bool>> triggered(std::make_shared<std::atomic<bool>>(false));

        thread_init_data data(
            util::detail::bind_front(&wake_timer_thread, thrd, newstate, newstate_ex, priority,
                self_id.noref(), triggered, retry_on_active),
            "wake_timer", priority, execution::thread_schedule_hint(),
            execution::thread_stacksize::small_, thread_schedule_state::suspended, true);

        thread_id_ref_type wake_id = invalid_thread_id;
        create_thread(scheduler, data, wake_id);

        PIKA_THROW_EXCEPTION(
            pika::error::invalid_status, "at_timer", "Timed suspension is currently not supported");
        // // create timer firing in correspondence with given time
        // using deadline_timer =
        //     asio::basic_waitable_timer<std::chrono::steady_clock>;

        // asio::io_context* s = get_default_timer_service();
        // PIKA_ASSERT(s);
        // deadline_timer t(*s, abs_time);

        // // let the timer invoke the set_state on the new (suspended) thread
        // t.async_wait([wake_id = std::move(wake_id), priority, retry_on_active](
        //                  std::error_code const& ec) {
        //     if (ec == std::make_error_code(std::errc::operation_canceled))
        //     {
        //            set_thread_state(wake_id.noref(), thread_schedule_state::pending,
        //                thread_restart_state::abort, priority,
        //                execution::thread_schedule_hint(), retry_on_active, throws);
        //     }
        //     else
        //     {
        //         set_thread_state(wake_id.noref(),
        //             thread_schedule_state::pending,
        //             thread_restart_state::timeout, priority,
        //             execution::thread_schedule_hint(), retry_on_active, throws);
        //     }
        // });

        // if (started != nullptr)
        // {
        //     started->store(true);
        // }

        // // this waits for the thread to be reactivated when the timer fired
        // // if it returns signaled the timer has been canceled, otherwise
        // // the timer fired and the wake_timer_thread above has been executed
        // thread_restart_state statex = get_self().yield(thread_result_type(
        //     thread_schedule_state::suspended, invalid_thread_id));

        // PIKA_ASSERT(statex == thread_restart_state::abort ||
        //     statex == thread_restart_state::timeout);

        // // NOLINTNEXTLINE(bugprone-branch-clone)
        // if (thread_restart_state::timeout != statex)    //-V601
        // {
        //     triggered->store(true);
        //     // wake_timer_thread has not been executed yet, cancel timer
        //     t.cancel();
        // }
        // else
        // {
        //     set_thread_state(
        //         thrd.noref(), newstate, newstate_ex, priority);
        // }

        return thread_result_type(thread_schedule_state::terminated, invalid_thread_id);
    }

    /// Set a timer to set the state of the given \a thread to the given
    /// new value after it expired (at the given time)
    thread_id_ref_type set_thread_state_timed(scheduler_base* scheduler,
        pika::chrono::steady_time_point const& abs_time, thread_id_type const& thrd,
        thread_schedule_state newstate, thread_restart_state newstate_ex,
        execution::thread_priority priority, execution::thread_schedule_hint schedulehint,
        std::atomic<bool>* started, bool retry_on_active, error_code& ec)
    {
        if (PIKA_UNLIKELY(!thrd))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id, "threads::detail::set_thread_state",
                "null thread id encountered");
            return invalid_thread_id;
        }

        // this creates a new thread which creates the timer and handles the
        // requested actions
        thread_init_data data(
            util::detail::bind(&at_timer, scheduler, abs_time.value(), thread_id_ref_type(thrd),
                newstate, newstate_ex, priority, started, retry_on_active),
            "at_timer (expire at)", priority, schedulehint, execution::thread_stacksize::small_,
            thread_schedule_state::pending, true);

        thread_id_ref_type newid = invalid_thread_id;
        create_thread(scheduler, data, newid, ec);    //-V601
        return newid;
    }
}    // namespace pika::threads::detail
