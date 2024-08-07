//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/debugging/print.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/execute.hpp>
#include <pika/execution/algorithms/schedule_from.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/executors/thread_pool_scheduler.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/print.hpp>
#include <pika/threading_base/register_thread.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_description.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace pika::debug::detail {
    // a debug level of zero disables messages with a level>0
    // a debug level of N shows messages with level 1..N
    template <int Level>
    static print_threshold<Level, 0> bps_deb("SBYPASS");
}    // namespace pika::debug::detail

namespace pika { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////
    struct thread_pool_scheduler_queue_bypass : thread_pool_scheduler
    {
        using thread_pool_scheduler::get_fallback_annotation;
        using thread_pool_scheduler::thread_pool_scheduler;
        using thread_pool_scheduler::operator==;
        using thread_pool_scheduler::operator!=;

        // Since this scheduler executes the task immediately on whichever thread is active
        // we do not support "with_priority" property
        // we do not support "with_hint" property

        // support with_stacksize property
        friend thread_pool_scheduler_queue_bypass tag_invoke(
            pika::execution::experimental::with_stacksize_t,
            thread_pool_scheduler_queue_bypass const& scheduler,
            pika::execution::thread_stacksize stacksize)
        {
            auto sched_with_stacksize = scheduler;
            sched_with_stacksize.stacksize_ = stacksize;
            return sched_with_stacksize;
        }

        friend pika::execution::thread_stacksize tag_invoke(
            pika::execution::experimental::get_stacksize_t,
            thread_pool_scheduler_queue_bypass const& scheduler)
        {
            return scheduler.stacksize_;
        }

        // support with_annotation property
        friend constexpr thread_pool_scheduler_queue_bypass tag_invoke(
            pika::execution::experimental::with_annotation_t,
            thread_pool_scheduler_queue_bypass const& scheduler, char const* annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ = annotation;
            return sched_with_annotation;
        }

        friend thread_pool_scheduler_queue_bypass tag_invoke(
            pika::execution::experimental::with_annotation_t,
            thread_pool_scheduler_queue_bypass const& scheduler, std::string annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ =
                pika::detail::store_function_annotation(PIKA_MOVE(annotation));
            return sched_with_annotation;
        }

        // support get_annotation property
        friend constexpr char const* tag_invoke(pika::execution::experimental::get_annotation_t,
            thread_pool_scheduler_queue_bypass const& scheduler) noexcept
        {
            return scheduler.annotation_;
        }

        template <typename F>
        void execute(F&& f, char const* fallback_annotation) const
        {
            using namespace pika::debug::detail;
            using namespace pika::threads::detail;
            using namespace pika::detail;

            if (get_self_id() != invalid_thread_id)
            {
                auto id = get_self_id();
                PIKA_DETAIL_DP(bps_deb<5>,
                    error(str<>("(in)Valid thread_id"), threadinfo<thread_id_type*>(&id)));
                f();
                return;
            }

            // which thread index are we in the current pool
            const std::int16_t thread_num = get_local_thread_num_tss();
            // create a description object with annotation
            thread_description desc(
                f, (fallback_annotation != nullptr) ? fallback_annotation : "Bypass");
            // create thread using 'pending_do_not_schedule' to bypass queue insertion
            thread_init_data data(make_thread_function_nullary(PIKA_FORWARD(F, f)), desc, priority_,
                thread_schedule_hint(thread_num), stacksize_,
                thread_schedule_state::pending_do_not_schedule, true /* run_now */
            );
            // set the pool's scheduler in the data
            data.scheduler_base = pool_->get_scheduler();

            // create the full thread object and set it as pending
            thread_id_ref_type id = invalid_thread_id;
            data.scheduler_base->create_thread(data, &id, pika::throws);
            data.initial_state = thread_schedule_state::pending;
            PIKA_DETAIL_DP(
                bps_deb<5>, debug(str<>("create_thread"), threadinfo<thread_id_ref_type*>(&id)));

            // get ready to perform a context switch on the thread and change its state
            thread_data* threaddata = get_thread_id_data(id);
            thread_state task_state = threaddata->get_state(std::memory_order_relaxed);
            thread_state active_state;

            // If no other thread has altered task state, set new state to active
            // (state change CANNOT happen on another thread because we just created the task)
            if (!threaddata->set_state_tagged(thread_schedule_state::active, task_state,
                    active_state, std::memory_order_relaxed))
            {
                throw std::runtime_error("Thread state cannot fail here");
            }

            this_thread::detail::agent_storage* context_storage =
                this_thread::detail::get_agent_storage();

            // invoke thread callable (call the actual function stored in the task)
            PIKA_DETAIL_DP(bps_deb<5>,
                debug(str<>("Execute"), threadinfo<thread_id_ref_type*>(&id), threaddata));
            thread_result_type task_return = (*threaddata)(context_storage);

            // did this task return a new task (ideally to be immediately switched into?)
            thread_id_ref_type next = PIKA_MOVE(task_return.second);
            if (next != nullptr)
            {
                PIKA_DETAIL_DP(bps_deb<5>,
                    debug(str<>("Next task"), threadinfo<thread_id_ref_type*>(&next), threaddata));
                // if next == id then we should just reschedule the task
                // but lets assert to find out how often this happens
                PIKA_ASSERT(next != id);

                auto* scheduler = get_thread_id_data(next)->get_scheduler_base();
                scheduler->schedule_thread(PIKA_MOVE(next),
                    thread_schedule_hint(static_cast<std::int16_t>(thread_num)), true,
                    thread_priority::boost);
                scheduler->do_some_work(thread_num);
            }

            if (task_return.first == thread_schedule_state::terminated)
            {
                // just exit without any fanfare, terminated cleanup will take care of threaddata
                PIKA_DETAIL_DP(bps_deb<5>,
                    debug(str<>("Terminated"), threadinfo<thread_id_ref_type*>(&id), threaddata));
                return;
            }

            // update our state with the new value returned from task execution
            task_state =
                thread_state(task_return.first, task_state.state_ex(), task_state.tag() + 1);

            // if the threaddata->state still matches active_state update to new state
            // (could be stolen/changed by another thread here?)
            if (threaddata->restore_state(
                    task_state, active_state, std::memory_order_relaxed, std::memory_order_relaxed))
            {
                switch (task_state.state())
                {
                case thread_schedule_state::pending_boost:
                    threaddata->set_state(thread_schedule_state::pending);
                    [[fallthrough]];
                case thread_schedule_state::pending:
                {
                    data.scheduler_base->schedule_thread(id, thread_schedule_hint(thread_num),
                        false /*allow_fallback*/, execution::thread_priority::high);
                }
                break;
                case thread_schedule_state::suspended:
                {
                    PIKA_DETAIL_DP(bps_deb<5>,
                        debug(str<>("AfterRestore"), "thread_schedule_state::suspended",
                            threadinfo<thread_id_ref_type*>(&id), threaddata));
                    // thread sits in thread map until resumed
                    return;
                }
                default: throw std::runtime_error("fix this thread state type");
                }
            }
            else
            {
                throw std::runtime_error(
                    "Should not be possible for another thread to have modified our state");
            }
        }

        template <typename F>
        friend void tag_invoke(execute_t, thread_pool_scheduler_queue_bypass const& sched, F&& f)
        {
            sched.execute(PIKA_FORWARD(F, f), sched.get_fallback_annotation());
        }

        friend sender<thread_pool_scheduler_queue_bypass> tag_invoke(
            schedule_t, thread_pool_scheduler_queue_bypass&& sched)
        {
            return {PIKA_MOVE(sched)};
        }

        friend sender<thread_pool_scheduler_queue_bypass> tag_invoke(
            schedule_t, thread_pool_scheduler_queue_bypass const& sched)
        {
            return {sched};
        }

        // We customize schedule_from to customize transfer. We want transfer to
        // take the annotation from the calling context of transfer if needed
        // and available. If we don't customize schedule_from the schedule
        // customization will be called much later by the schedule_from default
        // implementation, and the annotation may no longer come from the
        // calling context  of transfer.
        //
        // This updates the annotation of the scheduler with
        // get_fallback_annotation. If scheduler already has an annotation it
        // will simply set the same annotation. However, if scheduler doesn't
        // have an annotation set it will get it from the context calling
        // transfer if available, and otherwise fall back to the default
        // annotation. Once the scheduler annotation has been updated we
        // construct a sender from the default schedule_from implementation.
        //
        // TODO: Can we simply dispatch to the default implementation? This is
        // disabled with the P2300 reference implementation because we don't
        // want to use implementation details of it.
#if !defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend auto tag_invoke(schedule_from_t, thread_pool_scheduler_queue_bypass&& scheduler,
            Sender&& predecessor_sender)
        {
            return schedule_from_detail::schedule_from_sender<Sender,
                thread_pool_scheduler_queue_bypass>{PIKA_FORWARD(Sender, predecessor_sender),
                with_annotation(PIKA_MOVE(scheduler), scheduler.get_fallback_annotation())};
        }

        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend auto tag_invoke(schedule_from_t, thread_pool_scheduler_queue_bypass const& scheduler,
            Sender&& predecessor_sender)
        {
            return schedule_from_detail::schedule_from_sender<Sender,
                thread_pool_scheduler_queue_bypass>{PIKA_FORWARD(Sender, predecessor_sender),
                with_annotation(scheduler, scheduler.get_fallback_annotation())};
        }
#endif
        /// \endcond
    };
}}}    // namespace pika::execution::experimental
