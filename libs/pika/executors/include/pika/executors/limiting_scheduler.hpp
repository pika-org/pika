//  Copyright (c) 2023 ETH Zurich
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
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/synchronization/counting_semaphore.hpp>
#include <pika/threading_base/annotated_function.hpp>
#include <pika/threading_base/register_thread.hpp>
#include <pika/threading_base/thread_description.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace pika::execution::experimental {

    // -----------------------------------------------------------------
    // by convention the title is 7 chars (for alignment)
    // a debug level of N shows messages with level 1..N
    using namespace pika::debug::detail;
    template <int Level>
    static print_threshold<Level, 1> lsc_debug("SCLIMIT");

    template <typename WrappedScheduler>
    struct limiting_scheduler
    {
        using semaphore_type = pika::counting_semaphore<>;
        //
        constexpr limiting_scheduler() = delete;
        explicit limiting_scheduler(int max_n, WrappedScheduler on_scheduler)
          : internal_scheduler_(on_scheduler)
        {
            semaphore_ = std::make_shared<semaphore_type>(max_n);
            PIKA_DETAIL_DP(lsc_debug<5>, debug(str<>("construct"), max_n));
        }

        /// \cond NOINTERNAL
        bool operator==(limiting_scheduler const& rhs) const noexcept
        {
            return internal_scheduler_ == rhs.internal_scheduler_;
        }

        bool operator!=(limiting_scheduler const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        WrappedScheduler& get_internal_scheduler()
        {
            return internal_scheduler_;
        }

        template <typename F>
        void execute(F&& f) const
        {
            using namespace pika::debug::detail;
            PIKA_DETAIL_DP(lsc_debug<5>, debug(str<>("acquire")));
            semaphore_->acquire();
            auto semaphore_lambda = [f = std::forward<F>(f), sem = this->semaphore_]() mutable {
                PIKA_INVOKE(f, );
                PIKA_DETAIL_DP(lsc_debug<5>, debug(str<>("release")));
                sem->release();
            };
            pika::execution::experimental::execute(
                internal_scheduler_, PIKA_MOVE(semaphore_lambda));
        }

        template <typename F>
        friend void tag_invoke(execute_t, limiting_scheduler const& sched, F&& f)
        {
            execute(sched, PIKA_FORWARD(F, f));
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
                        os.scheduler.execute([receiver = PIKA_MOVE(os.receiver)]() mutable {
                            pika::execution::experimental::set_value(PIKA_MOVE(receiver));
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
            using is_sender = void;

            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures = pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t(),
                pika::execution::experimental::set_error_t(std::exception_ptr)>;

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver>
            tag_invoke(connect_t, sender&& s, Receiver&& receiver)
            {
                return {PIKA_MOVE(s.scheduler), PIKA_FORWARD(Receiver, receiver)};
            }

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver>
            tag_invoke(connect_t, sender const& s, Receiver&& receiver)
            {
                return {s.scheduler, PIKA_FORWARD(Receiver, receiver)};
            }

            struct env
            {
                PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

                friend std::decay_t<Scheduler> tag_invoke(
                    pika::execution::experimental::get_completion_scheduler_t<
                        pika::execution::experimental::set_value_t>,
                    env const& e) noexcept
                {
                    return e.scheduler;
                }
            };

            friend env tag_invoke(pika::execution::experimental::get_env_t, sender const& s)
            {
                return {s.scheduler};
            }
        };

        friend sender<limiting_scheduler> tag_invoke(schedule_t, limiting_scheduler&& sched)
        {
            return {PIKA_MOVE(sched)};
        }

        friend sender<limiting_scheduler> tag_invoke(schedule_t, limiting_scheduler const& sched)
        {
            return {sched};
        }

    private:
        WrappedScheduler internal_scheduler_;
        std::shared_ptr<semaphore_type> semaphore_;
        /// \endcond
    };
}    // namespace pika::execution::experimental
