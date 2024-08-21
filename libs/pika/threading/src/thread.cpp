//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>
#include <pika/assert.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/functional/bind.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/memory/intrusive_ptr.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/threading.hpp>
#include <pika/thread_support/unlock_guard.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/threading_base/thread_init_data.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#include <pika/timing/steady_clock.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <utility>

#if defined(__ANDROID__) || defined(ANDROID)
# include <cpu-features.h>
#endif
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.threading;
#endif

namespace pika {
    namespace detail {
        static thread_termination_handler_type thread_termination_handler;
    }

    void set_thread_termination_handler(thread_termination_handler_type f)
    {
        detail::thread_termination_handler = f;
    }

    thread::thread() noexcept
      : id_(pika::threads::detail::invalid_thread_id)
    {
    }

    thread::thread(thread&& rhs) noexcept
    {
        std::lock_guard l(rhs.mtx_);
        id_ = rhs.id_;
        rhs.id_ = threads::detail::invalid_thread_id;
    }

    thread& thread::operator=(thread&& rhs) noexcept
    {
        std::unique_lock l(mtx_);
        std::unique_lock l2(rhs.mtx_);
        if (joinable_locked())
        {
            l2.unlock();
            l.unlock();
            PIKA_THROW_EXCEPTION(
                pika::error::invalid_status, "thread::operator=", "destroying running thread");
        }
        id_ = rhs.id_;
        rhs.id_ = threads::detail::invalid_thread_id;
        return *this;
    }

    thread::~thread()
    {
        if (joinable())
        {
            if (detail::thread_termination_handler)
            {
                try
                {
                    PIKA_THROW_EXCEPTION(pika::error::invalid_status, "thread::~thread",
                        "destroying running thread");
                }
                catch (...)
                {
                    detail::thread_termination_handler(std::current_exception());
                }
            }
            else { std::terminate(); }
        }

        PIKA_ASSERT(id_ == threads::detail::invalid_thread_id);
    }

    void thread::swap(thread& rhs) noexcept
    {
        std::lock_guard l(mtx_);
        std::lock_guard l2(rhs.mtx_);
        std::swap(id_, rhs.id_);
    }

    static void run_thread_exit_callbacks()
    {
        threads::detail::thread_id_type id = pika::threads::detail::get_self_id();
        if (id == threads::detail::invalid_thread_id)
        {
            PIKA_THROW_EXCEPTION(pika::error::null_thread_id, "run_thread_exit_callbacks",
                "null thread id encountered");
        }
        threads::detail::run_thread_exit_callbacks(id);
        threads::detail::free_thread_exit_callbacks(id);
    }

    threads::detail::thread_result_type thread::thread_function_nullary(
        util::detail::unique_function<void()> const& func)
    {
        try
        {
            // Now notify our calling thread that we started execution.
            func();
        }
        catch (pika::thread_interrupted const&)
        {    //-V565
            /* swallow this exception */
        }
        catch (pika::exception const&)
        {
            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();

            // run all callbacks attached to the exit event for this thread
            run_thread_exit_callbacks();

            throw;    // rethrow any exception except 'thread_interrupted'
        }

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        // run all callbacks attached to the exit event for this thread
        run_thread_exit_callbacks();

        return threads::detail::thread_result_type(
            threads::detail::thread_schedule_state::terminated, threads::detail::invalid_thread_id);
    }

    thread::id thread::get_id() const noexcept { return id(native_handle()); }

    unsigned int thread::hardware_concurrency() noexcept
    {
        return pika::threads::detail::hardware_concurrency();
    }

    void thread::start_thread(
        threads::detail::thread_pool_base* pool, util::detail::unique_function<void()>&& func)
    {
        PIKA_ASSERT(pool);

        threads::detail::thread_init_data data(
            util::detail::one_shot(
                util::detail::bind(&thread::thread_function_nullary, PIKA_MOVE(func))),
            "thread::thread_function_nullary", execution::thread_priority::default_,
            execution::thread_schedule_hint(), execution::thread_stacksize::default_,
            threads::detail::thread_schedule_state::pending, true);

        // create the new thread, note that id_ is guaranteed to be valid
        // before the thread function is executed
        error_code ec(throwmode::lightweight);
        pool->create_thread(data, id_, ec);
        if (ec)
        {
            PIKA_THROW_EXCEPTION(pika::error::thread_resource_error, "thread::start_thread",
                "Could not create thread");
            return;
        }
    }

    static void resume_thread(threads::detail::thread_id_type const& id)
    {
        threads::detail::set_thread_state(id, threads::detail::thread_schedule_state::pending);
    }

    void thread::join()
    {
        std::unique_lock l(mtx_);

        if (!joinable_locked())
        {
            l.unlock();
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "thread::join",
                "trying to join a non joinable thread");
        }

        native_handle_type this_id = pika::threads::detail::get_self_id();
        if (this_id == id_)
        {
            l.unlock();
            PIKA_THROW_EXCEPTION(pika::error::thread_resource_error, "thread::join",
                "pika::thread: trying joining itself");
            return;
        }
        this_thread::interruption_point();

        // register callback function to be called when thread exits
        if (threads::detail::add_thread_exit_callback(
                id_.noref(), util::detail::bind_front(&resume_thread, this_id)))
        {
            // wait for thread to be terminated
            detail::unlock_guard ul(l);
            this_thread::suspend(threads::detail::thread_schedule_state::suspended, "thread::join");
        }

        detach_locked();    // invalidate this object
    }

    // extensions
    void thread::interrupt(bool flag) { threads::detail::interrupt_thread(native_handle(), flag); }

    bool thread::interruption_requested() const
    {
        return threads::detail::get_thread_interruption_requested(native_handle());
    }

    void thread::interrupt(thread::id id, bool flag)
    {
        threads::detail::interrupt_thread(id.id_, flag);
    }

    std::size_t thread::get_thread_data() const
    {
        return threads::detail::get_thread_data(native_handle());
    }
    std::size_t thread::set_thread_data(std::size_t data)
    {
        return threads::detail::set_thread_data(native_handle(), data);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread {

        void yield_to(thread::id id) noexcept
        {
            this_thread::suspend(threads::detail::thread_schedule_state::pending,
                id.native_handle(), "this_thread::yield_to");
        }

        void yield() noexcept
        {
            this_thread::suspend(
                threads::detail::thread_schedule_state::pending, "this_thread::yield");
        }

        thread::id get_id() noexcept { return thread::id(threads::detail::get_self_id()); }

        // extensions
        execution::thread_priority get_priority()
        {
            return threads::detail::get_thread_priority(threads::detail::get_self_id());
        }

        std::ptrdiff_t get_stack_size()
        {
            return threads::detail::get_stack_size(threads::detail::get_self_id());
        }

        void interruption_point()
        {
            threads::detail::interruption_point(threads::detail::get_self_id());
        }

        bool interruption_enabled()
        {
            return threads::detail::get_thread_interruption_enabled(threads::detail::get_self_id());
        }

        bool interruption_requested()
        {
            return threads::detail::get_thread_interruption_requested(
                pika::threads::detail::get_self_id());
        }

        void interrupt()
        {
            threads::detail::interrupt_thread(threads::detail::get_self_id());
            threads::detail::interruption_point(threads::detail::get_self_id());
        }

        void sleep_until(pika::chrono::steady_time_point const& abs_time)
        {
            this_thread::suspend(abs_time, "this_thread::sleep_until");
        }

        std::size_t get_thread_data()
        {
            return threads::detail::get_thread_data(threads::detail::get_self_id());
        }

        std::size_t set_thread_data(std::size_t data)
        {
            return threads::detail::set_thread_data(threads::detail::get_self_id(), data);
        }

        ///////////////////////////////////////////////////////////////////////
        disable_interruption::disable_interruption()
          : interruption_was_enabled_(interruption_enabled())
        {
            if (interruption_was_enabled_)
            {
                interruption_was_enabled_ = threads::detail::set_thread_interruption_enabled(
                    pika::threads::detail::get_self_id(), false);
            }
        }

        disable_interruption::~disable_interruption()
        {
            threads::detail::thread_self* p = threads::detail::get_self_ptr();
            if (p)
            {
                threads::detail::set_thread_interruption_enabled(
                    pika::threads::detail::get_self_id(), interruption_was_enabled_);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        restore_interruption::restore_interruption(disable_interruption& d)
          : interruption_was_enabled_(d.interruption_was_enabled_)
        {
            if (!interruption_was_enabled_)
            {
                interruption_was_enabled_ = threads::detail::set_thread_interruption_enabled(
                    pika::threads::detail::get_self_id(), true);
            }
        }

        restore_interruption::~restore_interruption()
        {
            threads::detail::thread_self* p = threads::detail::get_self_ptr();
            if (p)
            {
                threads::detail::set_thread_interruption_enabled(
                    pika::threads::detail::get_self_id(), interruption_was_enabled_);
            }
        }
    }    // namespace this_thread
}    // namespace pika
