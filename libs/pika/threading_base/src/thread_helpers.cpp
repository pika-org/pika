//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/threading_base/thread_helpers.hpp>

#include <pika/assert.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/modules/errors.hpp>
#ifdef PIKA_HAVE_VERIFY_LOCKS
# include <pika/lock_registration/detail/register_locks.hpp>
#endif
#include <pika/execution_base/this_thread.hpp>
#include <pika/threading_base/detail/reset_lco_description.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/scheduler_state.hpp>
#include <pika/threading_base/set_thread_state.hpp>
#include <pika/threading_base/set_thread_state_timed.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#include <pika/timing/steady_clock.hpp>
#include <pika/type_support/unused.hpp>

#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
# include <pika/debugging/backtrace.hpp>
# include <pika/threading_base/detail/reset_backtrace.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {

    ///////////////////////////////////////////////////////////////////////////
    thread_state set_thread_state(thread_id_type const& id, thread_schedule_state state,
        thread_restart_state stateex, execution::thread_priority priority, bool retry_on_active,
        error_code& ec)
    {
        if (&ec != &throws) ec = make_success_code();

        return set_thread_state(
            id, state, stateex, priority, execution::thread_schedule_hint(), retry_on_active, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_ref_type set_thread_state(thread_id_type const& id,
        pika::chrono::steady_time_point const& abs_time, std::atomic<bool>* timer_started,
        thread_schedule_state state, thread_restart_state stateex,
        execution::thread_priority priority, bool retry_on_active, error_code& ec)
    {
        return set_thread_state_timed(get_thread_id_data(id)->get_scheduler_base(), abs_time, id,
            state, stateex, priority, execution::thread_schedule_hint(), timer_started,
            retry_on_active, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_state get_thread_state(thread_id_type const& id, error_code& /* ec */)
    {
        return id ? get_thread_id_data(id)->get_state() :
                    thread_state(thread_schedule_state::terminated, thread_restart_state::unknown);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_phase(thread_id_type const& id, error_code& /* ec */)
    {
        return id ? get_thread_id_data(id)->get_thread_phase() : std::size_t(~0);
    }

    ///////////////////////////////////////////////////////////////////////////
    execution::thread_priority get_thread_priority(thread_id_type const& id, error_code& /* ec */)
    {
        return id ? get_thread_id_data(id)->get_priority() : execution::thread_priority::unknown;
    }

    std::ptrdiff_t get_stack_size(thread_id_type const& id, error_code& /* ec */)
    {
        return id ? get_thread_id_data(id)->get_stack_size() :
                    static_cast<std::ptrdiff_t>(execution::thread_stacksize::unknown);
    }

    void interrupt_thread(thread_id_type const& id, bool flag, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::interrupt_thread", "null thread id encountered");
            return;
        }

        if (&ec != &throws) ec = make_success_code();

        get_thread_id_data(id)->interrupt(flag);    // notify thread

        // Set thread state to pending. If the thread is currently active we do
        // not retry. The thread will either exit or hit an interruption_point.
        set_thread_state(id, thread_schedule_state::pending, thread_restart_state::abort,
            execution::thread_priority::normal, false, ec);
    }

    void interruption_point(thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::interruption_point", "null thread id encountered");
            return;
        }

        if (&ec != &throws) ec = make_success_code();

        get_thread_id_data(id)->interruption_point();    // notify thread
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_thread_interruption_enabled(thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROW_EXCEPTION(pika::error::null_thread_id,
                "pika::threads::detail::get_thread_interruption_enabled",
                "null thread id encountered");
            return false;
        }

        if (&ec != &throws) ec = make_success_code();

        return get_thread_id_data(id)->interruption_enabled();
    }

    bool set_thread_interruption_enabled(thread_id_type const& id, bool enable, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROW_EXCEPTION(pika::error::null_thread_id,
                "pika::threads::detail::get_thread_interruption_enabled",
                "null thread id encountered");
            return false;
        }

        if (&ec != &throws) ec = make_success_code();

        return get_thread_id_data(id)->set_interruption_enabled(enable);
    }

    bool get_thread_interruption_requested(thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::get_thread_interruption_requested",
                "null thread id encountered");
            return false;
        }

        if (&ec != &throws) ec = make_success_code();

        return get_thread_id_data(id)->interruption_requested();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_thread_data(thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::get_thread_data", "null thread id encountered");
            return 0;
        }

        return get_thread_id_data(id)->get_thread_data();
    }

    std::size_t set_thread_data(thread_id_type const& id, std::size_t data, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::set_thread_data", "null thread id encountered");
            return 0;
        }

        return get_thread_id_data(id)->set_thread_data(data);
    }

    ////////////////////////////////////////////////////////////////////////////
    static thread_local std::size_t continuation_recursion_count(0);

    std::size_t& get_continuation_recursion_count() noexcept
    {
        thread_self* self_ptr = get_self_ptr();
        if (self_ptr) { return self_ptr->get_continuation_recursion_count(); }
        return continuation_recursion_count;
    }

    void reset_continuation_recursion_count() noexcept { continuation_recursion_count = 0; }

    ///////////////////////////////////////////////////////////////////////////
    void run_thread_exit_callbacks(thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::run_thread_exit_callbacks", "null thread id encountered");
            return;
        }

        if (&ec != &throws) ec = make_success_code();

        get_thread_id_data(id)->run_thread_exit_callbacks();
    }

    bool add_thread_exit_callback(
        thread_id_type const& id, util::detail::function<void()> const& f, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::add_thread_exit_callback", "null thread id encountered");
            return false;
        }

        if (&ec != &throws) ec = make_success_code();

        return get_thread_id_data(id)->add_thread_exit_callback(f);
    }

    void free_thread_exit_callbacks(thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::add_thread_exit_callback", "null thread id encountered");
            return;
        }

        if (&ec != &throws) ec = make_success_code();

        get_thread_id_data(id)->free_thread_exit_callbacks();
    }

    ///////////////////////////////////////////////////////////////////////////
#ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    char const* get_thread_backtrace(thread_id_type const& id, error_code& ec)
#else
    debug::detail::backtrace const* get_thread_backtrace(thread_id_type const& id, error_code& ec)
#endif
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::get_thread_backtrace", "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws) ec = make_success_code();

        return get_thread_id_data(id)->get_backtrace();
    }

#ifdef PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
    char const* set_thread_backtrace(thread_id_type const& id, char const* bt, error_code& ec)
#else
    debug::detail::backtrace const* set_thread_backtrace(
        thread_id_type const& id, debug::detail::backtrace const* bt, error_code& ec)
#endif
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id,
                "pika::threads::detail::set_thread_backtrace", "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws) ec = make_success_code();

        return get_thread_id_data(id)->set_backtrace(bt);
    }

    threads::detail::thread_pool_base* get_pool(thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, pika::error::null_thread_id, "pika::threads::detail::get_pool",
                "null thread id encountered");
            return nullptr;
        }

        if (&ec != &throws) ec = make_success_code();

        return get_thread_id_data(id)->get_scheduler_base()->get_parent_pool();
    }
}    // namespace pika::threads::detail

namespace pika::this_thread {

    /// The function \a suspend will return control to the thread manager
    /// (suspends the current thread). It sets the new state of this thread
    /// to the thread state passed as the parameter.
    ///
    /// If the suspension was aborted, this function will throw a
    /// \a yield_aborted exception.
    pika::threads::detail::thread_restart_state suspend(
        threads::detail::thread_schedule_state state, threads::detail::thread_id_type nextid,
        detail::thread_description const& description, error_code& ec)
    {
        // let the thread manager do other things while waiting
        threads::detail::thread_self& self = threads::detail::get_self();

        // keep alive
        threads::detail::thread_id_ref_type id = self.get_thread_id();

        // handle interruption, if needed
        threads::detail::interruption_point(id.noref(), ec);
        if (ec) return pika::threads::detail::thread_restart_state::unknown;

        pika::threads::detail::thread_restart_state statex =
            pika::threads::detail::thread_restart_state::unknown;

        {
            // verify that there are no more registered locks for this OS-thread
#ifdef PIKA_HAVE_VERIFY_LOCKS
            util::verify_no_locks();
#endif
#ifdef PIKA_HAVE_THREAD_DESCRIPTION
            threads::detail::reset_lco_description desc(id.noref(), description, ec);
#else
            PIKA_UNUSED(description);
#endif
#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
            threads::detail::reset_backtrace bt(id, ec);
#endif
            // We might need to dispatch 'nextid' to it's correct scheduler
            // only if our current scheduler is the same, we should yield the id
            if (nextid &&
                get_thread_id_data(nextid)->get_scheduler_base() !=
                    get_thread_id_data(id)->get_scheduler_base())
            {
                auto* scheduler = get_thread_id_data(nextid)->get_scheduler_base();
                scheduler->schedule_thread(std::move(nextid), execution::thread_schedule_hint());
                statex = self.yield(
                    threads::detail::thread_result_type(state, threads::detail::invalid_thread_id));
            }
            else
            {
                statex = self.yield(threads::detail::thread_result_type(state, std::move(nextid)));
            }
        }

        // handle interruption, if needed
        threads::detail::interruption_point(id.noref(), ec);
        if (ec) return pika::threads::detail::thread_restart_state::unknown;

        // handle interrupt and abort
        if (statex == pika::threads::detail::thread_restart_state::abort)
        {
            PIKA_THROWS_IF(ec, pika::error::yield_aborted, "suspend",
                "thread({}, {}) aborted (yield returned wait_abort)", id.noref(),
                threads::detail::get_thread_description(id.noref()));
        }

        if (&ec != &throws) ec = make_success_code();

        return statex;
    }

    pika::threads::detail::thread_restart_state suspend(
        pika::chrono::steady_time_point const& abs_time, threads::detail::thread_id_type nextid,
        detail::thread_description const& description, error_code& ec)
    {
        // schedule a thread waking us up at_time
        threads::detail::thread_self& self = threads::detail::get_self();

        // keep alive
        threads::detail::thread_id_ref_type id = self.get_thread_id();

        // handle interruption, if needed
        threads::detail::interruption_point(id.noref(), ec);
        if (ec) return pika::threads::detail::thread_restart_state::unknown;

        // let the thread manager do other things while waiting
        pika::threads::detail::thread_restart_state statex =
            pika::threads::detail::thread_restart_state::unknown;

        {
#ifdef PIKA_HAVE_VERIFY_LOCKS
            // verify that there are no more registered locks for this OS-thread
            util::verify_no_locks();
#endif
#ifdef PIKA_HAVE_THREAD_DESCRIPTION
            threads::detail::reset_lco_description desc(id.noref(), description, ec);
#else
            PIKA_UNUSED(description);
#endif
#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
            threads::detail::reset_backtrace bt(id, ec);
#endif
            std::atomic<bool> timer_started(false);
            threads::detail::thread_id_ref_type timer_id =
                threads::detail::set_thread_state(id.noref(), abs_time, &timer_started,
                    threads::detail::thread_schedule_state::pending,
                    pika::threads::detail::thread_restart_state::timeout,
                    execution::thread_priority::boost, true, ec);
            if (ec) return pika::threads::detail::thread_restart_state::unknown;

            // We might need to dispatch 'nextid' to it's correct scheduler
            // only if our current scheduler is the same, we should yield the id
            if (nextid &&
                get_thread_id_data(nextid)->get_scheduler_base() !=
                    get_thread_id_data(id)->get_scheduler_base())
            {
                auto* scheduler = get_thread_id_data(nextid)->get_scheduler_base();
                scheduler->schedule_thread(std::move(nextid), execution::thread_schedule_hint());
                statex = self.yield(threads::detail::thread_result_type(
                    threads::detail::thread_schedule_state::suspended,
                    threads::detail::invalid_thread_id));
            }
            else
            {
                statex = self.yield(threads::detail::thread_result_type(
                    threads::detail::thread_schedule_state::suspended, std::move(nextid)));
            }

            if (statex != pika::threads::detail::thread_restart_state::timeout)
            {
                PIKA_ASSERT(statex == pika::threads::detail::thread_restart_state::abort ||
                    statex == pika::threads::detail::thread_restart_state::signaled);
                error_code ec1(throwmode::lightweight);    // do not throw
                pika::util::yield_while(
                    [&timer_started]() { return !timer_started.load(); }, "set_thread_state_timed");
                threads::detail::set_thread_state(timer_id.noref(),
                    threads::detail::thread_schedule_state::pending,
                    pika::threads::detail::thread_restart_state::abort,
                    execution::thread_priority::boost, true, ec1);
            }
        }

        // handle interruption, if needed
        threads::detail::interruption_point(id.noref(), ec);
        if (ec) return pika::threads::detail::thread_restart_state::unknown;

        // handle interrupt and abort
        if (statex == pika::threads::detail::thread_restart_state::abort)
        {
            PIKA_THROWS_IF(ec, pika::error::yield_aborted, "suspend_at",
                "thread({}, {}) aborted (yield returned wait_abort)", id.noref(),
                threads::detail::get_thread_description(id.noref()));
        }

        if (&ec != &throws) ec = make_success_code();

        return statex;
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::detail::thread_pool_base* get_pool(error_code& ec)
    {
        return threads::detail::get_pool(threads::detail::get_self_id(), ec);
    }

    std::ptrdiff_t get_available_stack_space()
    {
        threads::detail::thread_self* self = threads::detail::get_self_ptr();
        if (self) { return self->get_available_stack_space(); }

        return (std::numeric_limits<std::ptrdiff_t>::max)();
    }

    bool has_sufficient_stack_space(std::size_t space_needed)
    {
        if (nullptr == pika::threads::detail::get_self_ptr()) return false;

#if defined(PIKA_HAVE_THREADS_GET_STACK_POINTER)
        std::ptrdiff_t remaining_stack = get_available_stack_space();
        if (remaining_stack < 0)
        {
            PIKA_THROW_EXCEPTION(
                pika::error::out_of_memory, "has_sufficient_stack_space", "Stack overflow");
        }
        bool sufficient_stack_space = std::size_t(remaining_stack) >= space_needed;

        return sufficient_stack_space;
#else
        return true;
#endif
    }
}    // namespace pika::this_thread
