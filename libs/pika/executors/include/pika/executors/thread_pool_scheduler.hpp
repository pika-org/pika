//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/register_thread.hpp>

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
        void execute(F&& f) const
        {
            char const* annotation = annotation_ == nullptr ?
                traits::get_function_annotation<std::decay_t<F>>::call(f) :
                annotation_;

            threads::thread_init_data data(
                threads::make_thread_function_nullary(PIKA_FORWARD(F, f)),
                annotation, priority_, schedulehint_, stacksize_);
            threads::register_work(data, pool_);
        }

        template <typename Scheduler, typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Scheduler_, typename Receiver_>
            operation_state(Scheduler_&& scheduler, Receiver_&& receiver)
              : scheduler(PIKA_FORWARD(Scheduler_, scheduler))
              , receiver(PIKA_FORWARD(Receiver_, receiver))
            {
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
                            });
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
                return {PIKA_MOVE(s.scheduler), PIKA_FORWARD(Receiver, receiver)};
            }

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver> tag_invoke(
                connect_t, sender& s, Receiver&& receiver)
            {
                return {s.scheduler, PIKA_FORWARD(Receiver, receiver)};
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

        friend constexpr sender<thread_pool_scheduler> tag_invoke(
            schedule_t, thread_pool_scheduler&& sched)
        {
            return {PIKA_MOVE(sched)};
        }

        friend constexpr sender<thread_pool_scheduler> tag_invoke(
            schedule_t, thread_pool_scheduler const& sched)
        {
            return {sched};
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
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
