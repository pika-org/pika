//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>
#if defined(PIKA_HAVE_APEX)
# include <pika/threading_base/external_timer.hpp>
#endif
#include <pika/type_support/unused.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace pika::threads::detail {
    ///////////////////////////////////////////////////////////////////////////
    class thread_init_data
    {
    public:
        thread_init_data()
          : func()
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
          , description()
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
          , parent_id(nullptr)
          , parent_phase(0)
#endif
#ifdef PIKA_HAVE_APEX
          , timer_data(nullptr)
#endif
          , priority(execution::thread_priority::normal)
          , schedulehint()
          , stacksize(execution::thread_stacksize::default_)
          , initial_state(thread_schedule_state::pending)
          , run_now(false)
          , scheduler_base(nullptr)
        {
            if (initial_state == thread_schedule_state::staged)
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter,
                    "thread_init_data::thread_init_data",
                    "threads shouldn't have 'staged' as their initial state");
            }
        }

        thread_init_data& operator=(thread_init_data&& rhs) noexcept
        {
            func = std::move(rhs.func);
            priority = rhs.priority;
            schedulehint = rhs.schedulehint;
            stacksize = rhs.stacksize;
            initial_state = rhs.initial_state;
            run_now = rhs.run_now;
            scheduler_base = rhs.scheduler_base;
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
            description = std::move(rhs.description);
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
            parent_id = rhs.parent_id;
            parent_phase = rhs.parent_phase;
#endif
#ifdef PIKA_HAVE_APEX
            // PIKA_HAVE_APEX forces the PIKA_HAVE_THREAD_DESCRIPTION
            // and PIKA_HAVE_THREAD_PARENT_REFERENCE settings to be on
            timer_data = rhs.timer_data;
#endif
            return *this;
        }

        thread_init_data(thread_init_data&& rhs) noexcept
          : func(std::move(rhs.func))
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
          , description(std::move(rhs.description))
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
          , parent_id(rhs.parent_id)
          , parent_phase(rhs.parent_phase)
#endif
#ifdef PIKA_HAVE_APEX
          // PIKA_HAVE_APEX forces the PIKA_HAVE_THREAD_DESCRIPTION and
          // PIKA_HAVE_THREAD_PARENT_REFERENCE settings to be on
          , timer_data(pika::detail::external_timer::new_task(description, parent_id))
#endif
          , priority(rhs.priority)
          , schedulehint(rhs.schedulehint)
          , stacksize(rhs.stacksize)
          , initial_state(rhs.initial_state)
          , run_now(rhs.run_now)
          , scheduler_base(rhs.scheduler_base)
        {
        }

        template <typename F>
        thread_init_data(F&& f, ::pika::detail::thread_description const& desc,
            execution::thread_priority priority_ = execution::thread_priority::normal,
            execution::thread_schedule_hint os_thread = execution::thread_schedule_hint(),
            execution::thread_stacksize stacksize_ = execution::thread_stacksize::default_,
            thread_schedule_state initial_state_ = thread_schedule_state::pending,
            bool run_now_ = false,
            ::pika::threads::detail::scheduler_base* scheduler_base_ = nullptr)
          : func(std::forward<F>(f))
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
          , description(desc)
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
          , parent_id(nullptr)
          , parent_phase(0)
#endif
#ifdef PIKA_HAVE_APEX
          // PIKA_HAVE_APEX forces the PIKA_HAVE_THREAD_DESCRIPTION and
          // PIKA_HAVE_THREAD_PARENT_REFERENCE settings to be on
          , timer_data(pika::detail::external_timer::new_task(description, parent_id))
#endif
          , priority(priority_)
          , schedulehint(os_thread)
          , stacksize(stacksize_)
          , initial_state(initial_state_)
          , run_now(run_now_)
          , scheduler_base(scheduler_base_)
        {
#ifndef PIKA_HAVE_THREAD_DESCRIPTION
            PIKA_UNUSED(desc);
#endif
            if (initial_state == thread_schedule_state::staged)
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter,
                    "thread_init_data::thread_init_data",
                    "threads shouldn't have 'staged' as their initial state");
            }
        }

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
        ::pika::detail::thread_description get_description() const { return description; }
#else
        ::pika::detail::thread_description get_description() const
        {
            return ::pika::detail::thread_description("<unknown>");
        }
#endif

        thread_function_type func;

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
        ::pika::detail::thread_description description;
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
        thread_id_type parent_id;
        std::size_t parent_phase;
#endif
#ifdef PIKA_HAVE_APEX
        // PIKA_HAVE_APEX forces the PIKA_HAVE_THREAD_DESCRIPTION and
        // PIKA_HAVE_THREAD_PARENT_REFERENCE settings to be on
        std::shared_ptr<pika::detail::external_timer::task_wrapper> timer_data;
#endif

        execution::thread_priority priority;
        execution::thread_schedule_hint schedulehint;
        execution::thread_stacksize stacksize;
        thread_schedule_state initial_state;
        bool run_now;

        ::pika::threads::detail::scheduler_base* scheduler_base;
    };
}    // namespace pika::threads::detail
