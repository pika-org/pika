//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/coroutines/detail/coroutine_self.hpp>
#include <pika/coroutines/detail/tss.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/functional/detail/reset_function.hpp>
#include <pika/functional/unique_function.hpp>

#include <cstddef>
#include <utility>

namespace pika::threads::coroutines::detail {
    class stackless_coroutine
    {
    private:
        enum context_state
        {
            ctx_running,    // context running.
            ctx_ready,      // context at yield point.
            ctx_exited      // context is finished.
        };

        static constexpr std::ptrdiff_t default_stack_size = -1;

        bool running() const { return state_ == ctx_running; }

        bool exited() const { return state_ == ctx_exited; }

    public:
        friend struct coroutine_accessor;

        using thread_id_type = pika::threads::detail::thread_id;

        using result_type = std::pair<threads::detail::thread_schedule_state, thread_id_type>;
        using arg_type = pika::threads::detail::thread_restart_state;

        using functor_type = util::detail::unique_function<result_type(arg_type)>;

        stackless_coroutine(
            functor_type&& f, thread_id_type id, std::ptrdiff_t /*stack_size*/ = default_stack_size)
          : f_(std::move(f))
          , state_(ctx_ready)
          , id_(id)
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
          , phase_(0)
#endif
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
          , thread_data_(nullptr)
#else
          , thread_data_(0)
#endif
          , continuation_recursion_count_(0)
        {
        }

        ~stackless_coroutine()
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            delete_tss_storage(thread_data_);
#else
            thread_data_ = 0;
#endif
        }

        stackless_coroutine(stackless_coroutine const& src) = delete;
        stackless_coroutine& operator=(stackless_coroutine const& src) = delete;
        stackless_coroutine(stackless_coroutine&& src) = delete;
        stackless_coroutine& operator=(stackless_coroutine&& src) = delete;

        thread_id_type get_thread_id() const { return id_; }

#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t get_thread_phase() const { return phase_; }
#endif
        std::size_t get_thread_data() const
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            if (!thread_data_) return 0;
            return get_tss_thread_data(thread_data_);
#else
            return thread_data_;
#endif
        }

        std::size_t set_thread_data(std::size_t data)
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            return set_tss_thread_data(thread_data_, data);
#else
            std::size_t olddata = thread_data_;
            thread_data_ = data;
            return olddata;
#endif
        }

#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
        tss_storage* get_thread_tss_data(bool create_if_needed) const
        {
            if (!thread_data_ && create_if_needed) thread_data_ = create_tss_storage();
            return thread_data_;
        }
#endif

        void rebind(functor_type&& f, thread_id_type id)
        {
            PIKA_ASSERT(exited());

            f_ = std::move(f);
            id_ = id;

#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
            phase_ = 0;
#endif
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            PIKA_ASSERT(thread_data_ == nullptr);
#else
            PIKA_ASSERT(thread_data_ == 0);
#endif
            state_ = stackless_coroutine::ctx_ready;
        }

        void reset_tss()
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            delete_tss_storage(thread_data_);
#else
            thread_data_ = 0;
#endif
        }

        void reset()
        {
            PIKA_ASSERT(exited());

            util::detail::reset_function(f_);

#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
            phase_ = 0;
#endif
            id_.reset();
        }

    private:
        struct reset_on_exit
        {
            reset_on_exit(stackless_coroutine& self)
              : this_(self)
            {
                this_.state_ = stackless_coroutine::ctx_running;
            }

            ~reset_on_exit() { this_.state_ = stackless_coroutine::ctx_exited; }
            stackless_coroutine& this_;
        };
        friend struct reset_on_exit;

    public:
        PIKA_FORCEINLINE result_type operator()(arg_type arg = arg_type());

        explicit operator bool() const { return !exited(); }

        bool is_ready() const { return state_ == ctx_ready; }

        std::ptrdiff_t get_available_stack_space()
        {
            return (std::numeric_limits<std::ptrdiff_t>::max)();
        }

        std::size_t& get_continuation_recursion_count() { return continuation_recursion_count_; }

    protected:
        functor_type f_;
        context_state state_;
        thread_id_type id_;

#ifdef PIKA_HAVE_THREAD_PHASE_INFORMATION
        std::size_t phase_;
#endif
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
        mutable tss_storage* thread_data_;
#else
        mutable std::size_t thread_data_;
#endif
        std::size_t continuation_recursion_count_;
    };
}    // namespace pika::threads::coroutines::detail

////////////////////////////////////////////////////////////////////////////////
#include <pika/coroutines/detail/coroutine_stackless_self.hpp>

namespace pika::threads::coroutines::detail {
    PIKA_FORCEINLINE stackless_coroutine::result_type stackless_coroutine::operator()(arg_type arg)
    {
        PIKA_ASSERT(is_ready());

        result_type result(
            threads::detail::thread_schedule_state::terminated, threads::detail::invalid_thread_id);

        {
            coroutine_stackless_self self(this);
            reset_self_on_exit on_self_exit(&self, nullptr);

            {
                [[maybe_unused]] reset_on_exit on_exit{*this};

                result = f_(arg);    // invoke wrapped function

                // we always have to run to completion
                PIKA_ASSERT(result.first == threads::detail::thread_schedule_state::terminated);
            }

            reset_tss();
            reset();
        }

        return result;
    }
}    // namespace pika::threads::coroutines::detail
