//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/errors/throw_exception.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/logging.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_num_tss.hpp>

#include <pika/threading_base/detail/reset_lco_description.hpp>
#include <pika/threading_base/execution_agent.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/set_thread_state.hpp>
#include <pika/threading_base/thread_description.hpp>

#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#include <pika/debugging/backtrace.hpp>
#include <pika/threading_base/detail/reset_backtrace.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace pika { namespace threads {

    execution_agent::execution_agent(
        coroutines::detail::coroutine_impl* coroutine) noexcept
      : self_(coroutine)
    {
    }

    std::string execution_agent::description() const
    {
        thread_id_type id = self_.get_thread_id();
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROW_EXCEPTION(null_thread_id, "execution_agent::description",
                "null thread id encountered (is this executed on a "
                "pika-thread?)");
        }

        return pika::util::format(
            "{}: {}", id, get_thread_id_data(id)->get_description());
    }

    void execution_agent::yield(const char* desc)
    {
        do_yield(desc, pika::threads::thread_schedule_state::pending);
    }

    void execution_agent::yield_k(std::size_t k, const char* desc)
    {
        if (k < 4)    //-V112
        {
        }
        else if (k < 16)
        {
            PIKA_SMT_PAUSE;
        }
        else if (k < 32 || k & 1)    //-V112
        {
            do_yield(desc, pika::threads::thread_schedule_state::pending_boost);
        }
        else
        {
            do_yield(desc, pika::threads::thread_schedule_state::pending);
        }
    }

    void execution_agent::resume(const char* desc)
    {
        do_resume(desc, threads::thread_restart_state::signaled);
    }

    void execution_agent::abort(const char* desc)
    {
        do_resume(desc, threads::thread_restart_state::abort);
    }

    void execution_agent::suspend(const char* desc)
    {
        do_yield(desc, threads::thread_schedule_state::suspended);
    }

    void execution_agent::sleep_for(
        pika::chrono::steady_duration const& sleep_duration, const char* desc)
    {
        sleep_until(sleep_duration.from_now(), desc);
    }

    void execution_agent::sleep_until(
        pika::chrono::steady_time_point const& sleep_time, const char* desc)
    {
        // Just yield until time has passed by...
        auto now = std::chrono::steady_clock::now();

        // Note: we yield at least once to allow for other threads to
        // make progress in any case. We also use yield instead of yield_k
        // for the same reason.
        std::size_t k = 0;
        do
        {
            if (k < 32 || k & 1)
            {
                do_yield(
                    desc, pika::threads::thread_schedule_state::pending_boost);
            }
            else
            {
                do_yield(desc, pika::threads::thread_schedule_state::pending);
            }
            ++k;
            now = std::chrono::steady_clock::now();
        } while (now < sleep_time.value());
    }

#if defined(PIKA_HAVE_VERIFY_LOCKS)
    struct on_exit_reset_held_lock_data
    {
        on_exit_reset_held_lock_data()
          : data_(pika::util::get_held_locks_data())
        {
        }

        ~on_exit_reset_held_lock_data()
        {
            pika::util::set_held_locks_data(PIKA_MOVE(data_));
        }

        std::unique_ptr<pika::util::held_locks_data> data_;
    };
#else
    struct on_exit_reset_held_lock_data
    {
    };
#endif

    pika::threads::thread_restart_state execution_agent::do_yield(
        const char* desc, threads::thread_schedule_state state)
    {
        thread_id_ref_type id = self_.get_thread_id();    // keep alive
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROW_EXCEPTION(null_thread_id, "execution_agent::do_yield",
                "null thread id encountered (is this executed on a "
                "pika-thread?)");
        }

        // handle interruption, if needed
        thread_data* thrd_data = get_thread_id_data(id);
        PIKA_ASSERT(thrd_data);
        thrd_data->interruption_point();

        thrd_data->set_last_worker_thread_num(
            pika::get_local_worker_thread_num());

        threads::thread_restart_state statex =
            threads::thread_restart_state::unknown;

        {
#ifdef PIKA_HAVE_THREAD_DESCRIPTION
            threads::detail::reset_lco_description desc(
                id.noref(), util::thread_description(desc));
#endif
#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
            threads::detail::reset_backtrace bt(id);
#endif
            on_exit_reset_held_lock_data held_locks;
            PIKA_UNUSED(held_locks);

            PIKA_ASSERT(thrd_data->get_state().state() ==
                thread_schedule_state::active);
            PIKA_ASSERT(state != thread_schedule_state::active);
            statex = self_.yield(
                threads::thread_result_type(state, threads::invalid_thread_id));
            PIKA_ASSERT(get_thread_id_data(id)->get_state().state() ==
                thread_schedule_state::active);
        }

        // handle interruption, if needed
        thrd_data->interruption_point();

        // handle interrupt and abort
        if (statex == threads::thread_restart_state::abort)
        {
            PIKA_THROW_EXCEPTION(yield_aborted, desc,
                "thread({}) aborted (yield returned wait_abort)",
                description());
        }

        return statex;
    }

    void execution_agent::do_resume(
        const char* /* desc */, pika::threads::thread_restart_state statex)
    {
        threads::detail::set_thread_state(self_.get_thread_id(),
            thread_schedule_state::pending, statex, thread_priority::normal,
            thread_schedule_hint{}, false);
    }
}}    // namespace pika::threads
