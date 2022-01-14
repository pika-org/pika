//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>
#if defined(PIKA_HAVE_APEX)
#include <pika/threading_base/external_timer.hpp>
#endif
#include <pika/type_support/unused.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace pika { namespace threads {
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
          , parent_locality_id(0)
          , parent_id(nullptr)
          , parent_phase(0)
#endif
#ifdef PIKA_HAVE_APEX
          , timer_data(nullptr)
#endif
          , priority(thread_priority::normal)
          , schedulehint()
          , stacksize(thread_stacksize::default_)
          , initial_state(thread_schedule_state::pending)
          , run_now(false)
          , scheduler_base(nullptr)
        {
            if (initial_state == thread_schedule_state::staged)
            {
                PIKA_THROW_EXCEPTION(bad_parameter,
                    "thread_init_data::thread_init_data",
                    "threads shouldn't have 'staged' as their initial state");
            }
        }

        thread_init_data& operator=(thread_init_data&& rhs) noexcept
        {
            func = PIKA_MOVE(rhs.func);
            priority = rhs.priority;
            schedulehint = rhs.schedulehint;
            stacksize = rhs.stacksize;
            initial_state = rhs.initial_state;
            run_now = rhs.run_now;
            scheduler_base = rhs.scheduler_base;
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
            description = PIKA_MOVE(rhs.description);
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_id = rhs.parent_locality_id;
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
          : func(PIKA_MOVE(rhs.func))
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
          , description(PIKA_MOVE(rhs.description))
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
          , parent_locality_id(rhs.parent_locality_id)
          , parent_id(rhs.parent_id)
          , parent_phase(rhs.parent_phase)
#endif
#ifdef PIKA_HAVE_APEX
          // PIKA_HAVE_APEX forces the PIKA_HAVE_THREAD_DESCRIPTION and
          // PIKA_HAVE_THREAD_PARENT_REFERENCE settings to be on
          , timer_data(util::external_timer::new_task(
                description, parent_locality_id, parent_id))
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
        thread_init_data(F&& f, util::thread_description const& desc,
            thread_priority priority_ = thread_priority::normal,
            thread_schedule_hint os_thread = thread_schedule_hint(),
            thread_stacksize stacksize_ = thread_stacksize::default_,
            thread_schedule_state initial_state_ =
                thread_schedule_state::pending,
            bool run_now_ = false,
            policies::scheduler_base* scheduler_base_ = nullptr)
          : func(PIKA_FORWARD(F, f))
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
          , description(desc)
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
          , parent_locality_id(0)
          , parent_id(nullptr)
          , parent_phase(0)
#endif
#ifdef PIKA_HAVE_APEX
          // PIKA_HAVE_APEX forces the PIKA_HAVE_THREAD_DESCRIPTION and
          // PIKA_HAVE_THREAD_PARENT_REFERENCE settings to be on
          , timer_data(util::external_timer::new_task(
                description, parent_locality_id, parent_id))
#endif
          , priority(priority_)
          , schedulehint(os_thread)
          , stacksize(stacksize_)
          , initial_state(initial_state_)
          , run_now(run_now_)
          , scheduler_base(scheduler_base_)
        {
            PIKA_UNUSED(desc);

            if (initial_state == thread_schedule_state::staged)
            {
                PIKA_THROW_EXCEPTION(bad_parameter,
                    "thread_init_data::thread_init_data",
                    "threads shouldn't have 'staged' as their initial state");
            }
        }

        threads::thread_function_type func;

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
        util::thread_description description;
#endif
#if defined(PIKA_HAVE_THREAD_PARENT_REFERENCE)
        std::uint32_t parent_locality_id;
        threads::thread_id_type parent_id;
        std::size_t parent_phase;
#endif
#ifdef PIKA_HAVE_APEX
        // PIKA_HAVE_APEX forces the PIKA_HAVE_THREAD_DESCRIPTION and
        // PIKA_HAVE_THREAD_PARENT_REFERENCE settings to be on
        std::shared_ptr<util::external_timer::task_wrapper> timer_data;
#endif

        thread_priority priority;
        thread_schedule_hint schedulehint;
        thread_stacksize stacksize;
        thread_schedule_state initial_state;
        bool run_now;

        policies::scheduler_base* scheduler_base;
    };
}}    // namespace pika::threads
