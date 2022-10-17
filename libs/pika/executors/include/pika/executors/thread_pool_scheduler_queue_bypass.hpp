//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/execute.hpp>
#include <pika/execution/algorithms/schedule_from.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/executors/thread_pool_scheduler_queue_bypass.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/register_thread.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_description.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////
    using namespace pika::threads::detail;
    class switch_status_background
    {
    public:
        switch_status_background(thread_id_ref_type const& t, thread_state prev_state)
          : thread_(t)
          , prev_state_(prev_state)
          , next_thread_id_(nullptr)
          , need_restore_state_(get_thread_id_data(thread_)->set_state_tagged(
                thread_schedule_state::active, prev_state_, orig_state_, std::memory_order_relaxed))
        {
        }

        ~switch_status_background()
        {
            if (need_restore_state_) { store_state(prev_state_); }
        }

        bool is_valid() const { return need_restore_state_; }

        // allow to change the state the thread will be switched to after
        // execution
        thread_state operator=(thread_result_type&& new_state)
        {
            prev_state_ =
                thread_state(new_state.first, prev_state_.state_ex(), prev_state_.tag() + 1);
            next_thread_id_ = PIKA_MOVE(new_state.second);
            return prev_state_;
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other worker-thread is started to execute this
        // pika-thread in the meantime.
        thread_schedule_state get_previous() const { return prev_state_.state(); }

        // This restores the previous state, while making sure that the
        // original state has not been changed since we started executing this
        // thread. The function returns true if the state has been set, false
        // otherwise.
        bool store_state(thread_state& newstate)
        {
            disable_restore();
            if (get_thread_id_data(thread_)->restore_state(
                    prev_state_, orig_state_, std::memory_order_relaxed, std::memory_order_relaxed))
            {
                newstate = prev_state_;
                return true;
            }
            return false;
        }

        // disable default handling in destructor
        void disable_restore() { need_restore_state_ = false; }

        thread_id_ref_type const& get_next_thread() const { return next_thread_id_; }

        thread_id_ref_type move_next_thread() { return PIKA_MOVE(next_thread_id_); }

    private:
        thread_id_ref_type const& thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        thread_id_ref_type next_thread_id_;
        bool need_restore_state_;
    };

    struct thread_pool_scheduler_queue_bypass : thread_pool_scheduler
    {
        using thread_pool_scheduler::get_fallback_annotation;
        using thread_pool_scheduler::thread_pool_scheduler;
        using thread_pool_scheduler::operator==;
        using thread_pool_scheduler::operator!=;

        // support with_priority property
        friend thread_pool_scheduler_queue_bypass tag_invoke(
            pika::execution::experimental::with_priority_t,
            thread_pool_scheduler_queue_bypass const& scheduler,
            pika::execution::thread_priority priority)
        {
            auto sched_with_priority = scheduler;
            sched_with_priority.priority_ = priority;
            return sched_with_priority;
        }

        friend pika::execution::thread_priority tag_invoke(
            pika::execution::experimental::get_priority_t,
            thread_pool_scheduler_queue_bypass const& scheduler)
        {
            return scheduler.priority_;
        }

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

        // support with_hint property
        friend thread_pool_scheduler_queue_bypass tag_invoke(
            pika::execution::experimental::with_hint_t,
            thread_pool_scheduler_queue_bypass const& scheduler,
            pika::execution::thread_schedule_hint hint)
        {
            auto sched_with_hint = scheduler;
            sched_with_hint.schedulehint_ = hint;
            return sched_with_hint;
        }

        friend pika::execution::thread_schedule_hint tag_invoke(
            pika::execution::experimental::get_hint_t,
            thread_pool_scheduler_queue_bypass const& scheduler)
        {
            return scheduler.schedulehint_;
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
#if 0
            pika::detail::thread_description desc(f, fallback_annotation);
            threads::detail::thread_init_data data(
                threads::detail::make_thread_function_nullary(
                    PIKA_FORWARD(F, f)),
                desc, priority_, schedulehint_, stacksize_);
            threads::detail::register_work(data, pool_);
#else
            using namespace pika::threads::detail;
            using namespace pika::detail;
            // which thread index are we
            const std::int16_t thread_num = get_local_thread_num_tss();
            thread_description desc(f, fallback_annotation);

            // create thread using pending_do_not_scheduleto bypass queue insertion
            thread_init_data data(make_thread_function_nullary(PIKA_FORWARD(F, f)), desc, priority_,
                thread_schedule_hint(thread_num), stacksize_,
                thread_schedule_state::pending_do_not_schedule, true /* run_now */
            );
            data.scheduler_base = pool_->get_scheduler();

            // threads::detail::register_work(data, pool_);
            thread_id_ref_type id = invalid_thread_id;
            pika::error_code ec = throws;
            data.scheduler_base->create_thread(data, &id, ec);
            data.initial_state = thread_schedule_state::pending;

            thread_data* threaddata = get_thread_id_data(id);
            thread_state state = threaddata->get_state(std::memory_order_relaxed);

            switch_status_background thrd_stat(id, state);

            if (PIKA_LIKELY(thrd_stat.is_valid() &&
                    thrd_stat.get_previous() == thread_schedule_state::pending))
            {
                pika::execution::this_thread::detail::agent_storage* context_storage =
                    pika::execution::this_thread::detail::get_agent_storage();

                // invoke thread callable
                thrd_stat = (*threaddata)(context_storage);

                thread_id_ref_type next = thrd_stat.move_next_thread();
                if (next != nullptr && next != id)
                {
                    thread_id_ref_type next_thrd;
                    if (next_thrd == nullptr) { next_thrd = PIKA_MOVE(next); }
                    else
                    {
                        auto* scheduler = get_thread_id_data(next)->get_scheduler_base();
                        scheduler->schedule_thread(PIKA_MOVE(next),
                            execution::thread_schedule_hint(static_cast<std::int16_t>(thread_num)),
                            true);
                        scheduler->do_some_work(thread_num);
                    }
                }
                thrd_stat.store_state(state);
                //
                switch (state.state())
                {
                case thread_schedule_state::pending_boost:
                    // might happen? reset to pending and schedule it
                    threaddata->set_state(thread_schedule_state::pending);
                    // fallthrough
                case thread_schedule_state::pending:
                    data.scheduler_base->schedule_thread(id, thread_schedule_hint(thread_num),
                        false /*allow_fallback*/, execution::thread_priority::high);
                    break;
                case thread_schedule_state::terminated:
                    // thread goes into terminated items and gets cleaned up
                case thread_schedule_state::suspended:
                    // thread sits in thread map until resumed
                    return;
                default: throw std::runtime_error("fix this thread state type");
                }
            }
#endif
        }

        template <typename F>
        friend void tag_invoke(execute_t, thread_pool_scheduler_queue_bypass const& sched, F&& f)
        {
            sched.execute(PIKA_FORWARD(F, f), sched.get_fallback_annotation());
        }
        /*
        template <typename Scheduler, typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            char const* fallback_annotation;

            template <typename Scheduler_, typename Receiver_>
            operation_state(Scheduler_&& scheduler, Receiver_&& receiver,
                char const* fallback_annotation)
              : scheduler(PIKA_FORWARD(Scheduler_, scheduler))
              , receiver(PIKA_FORWARD(Receiver_, receiver))
              , fallback_annotation(fallback_annotation)
            {
                PIKA_ASSERT(fallback_annotation != nullptr);
            }

            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            friend void tag_invoke(start_t, operation_state& os) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        os.scheduler.execute(
                            [receiver = PIKA_MOVE(os.receiver)]() mutable {
                                pika::execution::experimental::set_value(
                                    PIKA_MOVE(receiver));
                            },
                            os.fallback_annotation);
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            PIKA_MOVE(os.receiver), PIKA_MOVE(ep));
                    });
            }
        };

        template <typename Scheduler>
        struct sender
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

            // We explicitly get the fallback annotation when constructing the
            // sender so that if the scheduler has no annotation, the annotation
            // is instead taken from the context creating the sender, not from
            // the context when the task is actually spawned (if different).
            char const* fallback_annotation =
                scheduler.get_fallback_annotation();

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures =
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_value_t(),
                    pika::execution::experimental::set_error_t(
                        std::exception_ptr)>;

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver> tag_invoke(
                connect_t, sender&& s, Receiver&& receiver)
            {
                return {PIKA_MOVE(s.scheduler),
                    PIKA_FORWARD(Receiver, receiver), s.fallback_annotation};
            }

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver> tag_invoke(
                connect_t, sender& s, Receiver&& receiver)
            {
                return {s.scheduler, PIKA_FORWARD(Receiver, receiver),
                    s.fallback_annotation};
            }

            template <typename CPO,
                PIKA_CONCEPT_REQUIRES_(std::is_same_v<CPO,
                    pika::execution::experimental::set_value_t>)>
            friend constexpr auto tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<CPO>,
                sender const& s) noexcept
            {
                return s.scheduler;
            }
        };
*/
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
