//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

/*
 * Currently asio can, in some cases. call copy constructors and
 * operator= from different threads, even if in the
 * one-thread-per-service model. (i.e. from the resolver thread)
 * This will be corrected in future versions, but for now
 * we will play it safe and use an atomic count. The overhead shouldn't
 * be big.
 */
#include <pika/config.hpp>

// This needs to be first for building on Macs
#include <pika/coroutines/detail/context_impl.hpp>

#include <pika/assert.hpp>
#include <pika/coroutines/detail/swap_context.hpp>    //for swap hints
#include <pika/coroutines/detail/tss.hpp>
#include <pika/coroutines/thread_id_type.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
#define PIKA_COROUTINE_NUM_ALL_HEAPS                                                               \
    (PIKA_COROUTINE_NUM_HEAPS + PIKA_COROUTINE_NUM_HEAPS / 2 + PIKA_COROUTINE_NUM_HEAPS / 4 +      \
        PIKA_COROUTINE_NUM_HEAPS / 4) /**/

namespace pika::threads::coroutines::detail {
    constexpr std::ptrdiff_t const default_stack_size = -1;

    template <typename CoroutineImpl>
    class context_base : public default_context_impl<CoroutineImpl>
    {
        using base_type = default_context_impl<CoroutineImpl>;

    public:
        using deleter_type = void(context_base const*);
        using thread_id_type = pika::threads::detail::thread_id;

        context_base(std::ptrdiff_t stack_size, thread_id_type id)
          : base_type(stack_size)
          , m_caller()
          , m_state(ctx_ready)
          , m_exit_state(ctx_exit_not_requested)
          , m_exit_status(ctx_not_exited)
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
          , m_phase(0)
#endif
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
          , m_thread_data(nullptr)
#else
          , m_thread_data(0)
#endif
          , m_type_info()
          , m_thread_id(id)
          , continuation_recursion_count_(0)
        {
        }

        void reset_tss()
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            delete_tss_storage(m_thread_data);
#else
            m_thread_data = 0;
#endif
        }

        void reset()
        {
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
            m_phase = 0;
#endif
            m_thread_id.reset();
        }

#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t phase() const { return m_phase; }
#endif

        thread_id_type get_thread_id() const { return m_thread_id; }

        /*
         * Returns true if the context is runnable.
         */
        bool is_ready() const { return m_state == ctx_ready; }

        bool running() const { return m_state == ctx_running; }

        bool exited() const { return m_state == ctx_exited; }

        void init() { base_type::init(); }

        // Resume coroutine.
        // Pre:  The coroutine must be ready.
        // Post: The coroutine relinquished control. It might be ready
        //       or exited.
        // Throws:- 'abnormal_exit' if the coroutine was exited by another
        //          uncaught exception.
        // Note, it guarantees that the coroutine is resumed. Can throw only
        // on return.
        void invoke()
        {
            base_type::init();
            PIKA_ASSERT(is_ready());
            do_invoke();

            if (m_exit_status != ctx_not_exited)
            {
                if (m_exit_status == ctx_exited_return) return;
                if (m_exit_status == ctx_exited_abnormally)
                {
                    PIKA_ASSERT(m_type_info);
                    std::rethrow_exception(m_type_info);
                }
                PIKA_ASSERT_MSG(false, "unknown exit status");
            }
        }

        // Put coroutine in ready state and relinquish control
        // to caller until resumed again.
        // Pre:  Coroutine is running.
        //       Exit not pending.
        //       Operations not pending.
        // Post: Coroutine is running.
        // Throws: exit_exception, if exit is pending *after* it has been
        //         resumed.
        void yield()
        {
            // Misbehaved threads may try to yield while handling an exception.
            // This is dangerous if the thread can migrate to other worker
            // threads since the count for std::uncaught_exceptions may become
            // inconsistent (including negative). If at any point in the future
            // there is a legitimate use case for yielding with uncaught
            // exceptions this assertion can be revisited, but until then we
            // prefer to be strict about it.
            PIKA_ASSERT(std::uncaught_exceptions() == 0);

            //prevent infinite loops
            PIKA_ASSERT(m_exit_state < ctx_exit_signaled);
            PIKA_ASSERT(running());

            m_state = ctx_ready;
#if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            this->start_yield_fiber(&this->asan_fake_stack, m_caller);
#endif
            do_yield();

#if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            this->finish_yield_fiber(this->asan_fake_stack);
#endif
            m_exit_status = ctx_not_exited;

            PIKA_ASSERT(running());
        }

        // Nothrow.
        ~context_base() noexcept
        {
            PIKA_ASSERT(!running());
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
            PIKA_ASSERT(exited() || (is_ready() && m_phase == 0));
#else
            PIKA_ASSERT(exited() || is_ready());
#endif
            m_thread_id.reset();
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            delete_tss_storage(m_thread_data);
#else
            m_thread_data = 0;
#endif
        }

        std::size_t get_thread_data() const
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            if (!m_thread_data) return 0;
            return get_tss_thread_data(m_thread_data);
#else
            return m_thread_data;
#endif
        }

        std::size_t set_thread_data(std::size_t data)
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            return set_tss_thread_data(m_thread_data, data);
#else
            std::size_t olddata = m_thread_data;
            m_thread_data = data;
            return olddata;
#endif
        }

#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
        tss_storage* get_thread_tss_data(bool create_if_needed) const
        {
            if (!m_thread_data && create_if_needed) m_thread_data = create_tss_storage();
            return m_thread_data;
        }
#endif

        std::size_t& get_continuation_recursion_count() { return continuation_recursion_count_; }

    public:
        // global coroutine state
        enum context_state
        {
            ctx_running = 0,    // context running.
            ctx_ready,          // context at yield point.
            ctx_exited          // context is finished.
        };

    protected:
        // exit request state
        enum context_exit_state
        {
            ctx_exit_not_requested = 0,    // exit not requested.
            ctx_exit_pending,              // exit requested.
            ctx_exit_signaled              // exit request delivered.
        };

        // exit status
        enum context_exit_status
        {
            ctx_not_exited,
            ctx_exited_return,       // process exited by return.
            ctx_exited_abnormally    // process exited uncleanly.
        };

        void rebind_base(thread_id_type id)
        {
            PIKA_ASSERT(!running());

            m_thread_id = id;
            m_state = ctx_ready;
            m_exit_state = ctx_exit_not_requested;
            m_exit_status = ctx_not_exited;
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
            PIKA_ASSERT(m_phase == 0);
#endif
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            PIKA_ASSERT(m_thread_data == nullptr);
#else
            PIKA_ASSERT(m_thread_data == 0);
#endif
            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
            m_type_info = std::exception_ptr();
        }

        // Nothrow.
        void do_return(context_exit_status status, std::exception_ptr&& info) noexcept
        {
            PIKA_ASSERT(status != ctx_not_exited);
            PIKA_ASSERT(m_state == ctx_running);
            m_type_info = std::move(info);
            m_state = ctx_exited;
            m_exit_status = status;
#if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            this->start_yield_fiber(&this->asan_fake_stack, m_caller);
#endif

            do_yield();
        }

    protected:
        // Nothrow.
        void do_yield() noexcept { swap_context(*this, m_caller, detail::yield_hint()); }

        // Nothrow.
        void do_invoke() noexcept
        {
            PIKA_ASSERT(is_ready());
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
            ++m_phase;
#endif
            m_state = ctx_running;

#if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            this->start_switch_fiber(&this->asan_fake_stack);
#endif

            swap_context(m_caller, *this, detail::invoke_hint());

#if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            this->finish_switch_fiber(this->asan_fake_stack, m_caller);
#endif
        }

        using ctx_type = typename base_type::context_impl_base;
        ctx_type m_caller;

        context_state m_state;
        context_exit_state m_exit_state;
        context_exit_status m_exit_status;
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t m_phase;
#endif
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
        mutable detail::tss_storage* m_thread_data;
#else
        mutable std::size_t m_thread_data;
#endif

        // This is used to generate a meaningful exception trace.
        std::exception_ptr m_type_info;
        thread_id_type m_thread_id;

        std::size_t continuation_recursion_count_;
    };
}    // namespace pika::threads::coroutines::detail
