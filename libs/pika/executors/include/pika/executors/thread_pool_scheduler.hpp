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
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/register_thread.hpp>
#include <pika/threading_base/thread_description.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    struct thread_pool_scheduler
    {
        constexpr thread_pool_scheduler() = default;
        explicit thread_pool_scheduler(pika::threads::thread_pool_base* pool)
          : pool_(pool)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(thread_pool_scheduler const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ &&
                schedulehint_ == rhs.schedulehint_;
        }

        bool operator!=(thread_pool_scheduler const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        pika::threads::thread_pool_base* get_thread_pool()
        {
            PIKA_ASSERT(pool_);
            return pool_;
        }

        // support with_priority property
        friend thread_pool_scheduler tag_invoke(
            pika::execution::experimental::with_priority_t,
            thread_pool_scheduler const& scheduler,
            pika::threads::thread_priority priority)
        {
            auto sched_with_priority = scheduler;
            sched_with_priority.priority_ = priority;
            return sched_with_priority;
        }

        friend pika::threads::thread_priority tag_invoke(
            pika::execution::experimental::get_priority_t,
            thread_pool_scheduler const& scheduler)
        {
            return scheduler.priority_;
        }

        // support with_stacksize property
        friend thread_pool_scheduler tag_invoke(
            pika::execution::experimental::with_stacksize_t,
            thread_pool_scheduler const& scheduler,
            pika::threads::thread_stacksize stacksize)
        {
            auto sched_with_stacksize = scheduler;
            sched_with_stacksize.stacksize_ = stacksize;
            return sched_with_stacksize;
        }

        friend pika::threads::thread_stacksize tag_invoke(
            pika::execution::experimental::get_stacksize_t,
            thread_pool_scheduler const& scheduler)
        {
            return scheduler.stacksize_;
        }

        // support with_hint property
        friend thread_pool_scheduler tag_invoke(
            pika::execution::experimental::with_hint_t,
            thread_pool_scheduler const& scheduler,
            pika::threads::thread_schedule_hint hint)
        {
            auto sched_with_hint = scheduler;
            sched_with_hint.schedulehint_ = hint;
            return sched_with_hint;
        }

        friend pika::threads::thread_schedule_hint tag_invoke(
            pika::execution::experimental::get_hint_t,
            thread_pool_scheduler const& scheduler)
        {
            return scheduler.schedulehint_;
        }

        // support with_annotation property
        friend constexpr thread_pool_scheduler tag_invoke(
            pika::execution::experimental::with_annotation_t,
            thread_pool_scheduler const& scheduler, char const* annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ = annotation;
            return sched_with_annotation;
        }

        friend thread_pool_scheduler tag_invoke(
            pika::execution::experimental::with_annotation_t,
            thread_pool_scheduler const& scheduler, std::string annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ =
                pika::detail::store_function_annotation(PIKA_MOVE(annotation));
            return sched_with_annotation;
        }

        // support get_annotation property
        friend constexpr char const* tag_invoke(
            pika::execution::experimental::get_annotation_t,
            thread_pool_scheduler const& scheduler) noexcept
        {
            return scheduler.annotation_;
        }

        template <typename F>
        void execute(F&& f, char const* fallback_annotation) const
        {
            pika::util::thread_description desc(f, fallback_annotation);
            threads::thread_init_data data(
                threads::make_thread_function_nullary(PIKA_FORWARD(F, f)), desc,
                priority_, schedulehint_, stacksize_);
            threads::register_work(data, pool_);
        }

        template <typename F>
        friend void tag_invoke(
            execute_t, thread_pool_scheduler const& sched, F&& f)
        {
            sched.execute(PIKA_FORWARD(F, f), sched.get_fallback_annotation());
        }

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
                sender const& s)
            {
                return s.scheduler;
            }
        };

        friend sender<thread_pool_scheduler> tag_invoke(
            schedule_t, thread_pool_scheduler&& sched)
        {
            return {PIKA_MOVE(sched)};
        }

        friend sender<thread_pool_scheduler> tag_invoke(
            schedule_t, thread_pool_scheduler const& sched)
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
        // TODO: Can we simply dispatch to the default implementation?
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend auto tag_invoke(schedule_from_t,
            thread_pool_scheduler&& scheduler, Sender&& predecessor_sender)
        {
            return schedule_from_detail::schedule_from_sender<Sender,
                thread_pool_scheduler>{PIKA_FORWARD(Sender, predecessor_sender),
                with_annotation(
                    PIKA_MOVE(scheduler), scheduler.get_fallback_annotation())};
        }

        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend auto tag_invoke(schedule_from_t,
            thread_pool_scheduler const& scheduler, Sender&& predecessor_sender)
        {
            return schedule_from_detail::schedule_from_sender<Sender,
                thread_pool_scheduler>{PIKA_FORWARD(Sender, predecessor_sender),
                with_annotation(
                    scheduler, scheduler.get_fallback_annotation())};
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        char const* get_fallback_annotation() const
        {
            // Scheduler annotations have priority
            if (annotation_)
            {
                return annotation_;
            }

            // Next is the annotation from the current context scheduling work
            pika::threads::thread_id_type id = pika::threads::get_self_id();
            if (id)
            {
                pika::util::thread_description desc =
                    pika::threads::get_thread_description(id);
                if (desc.kind() ==
                    pika::util::thread_description::data_type_description)
                {
                    return desc.get_description();
                }
            }

            // If there are no annotations in the scheduler or scheduling
            // context, use "<unknown>". Explicitly do not use nullptr here to
            // avoid thread_description taking the current annotation from the
            // spawning context.
            return "<unknown>";
        }

        pika::threads::thread_pool_base* pool_ =
            pika::threads::detail::get_self_or_default_pool();
        pika::threads::thread_priority priority_ =
            pika::threads::thread_priority::normal;
        pika::threads::thread_stacksize stacksize_ =
            pika::threads::thread_stacksize::small_;
        pika::threads::thread_schedule_hint schedulehint_{};
        char const* annotation_ = nullptr;
        /// \endcond
    };
}}}    // namespace pika::execution::experimental
