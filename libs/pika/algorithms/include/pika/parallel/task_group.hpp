//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file task_group.hpp

#pragma once

#include <pika/local/config.hpp>

#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/errors/exception_list.hpp>
#include <pika/execution_base/execution.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/executors/parallel_executor.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/modules/memory.hpp>
#include <pika/serialization/serialization_fwd.hpp>
#include <pika/synchronization/latch.hpp>
#include <pika/type_support/unused.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    class task_group
    {
    public:
        PIKA_EXPORT task_group();
        PIKA_EXPORT ~task_group();

    private:
        struct on_exit
        {
            PIKA_EXPORT explicit on_exit(task_group& tg);
            PIKA_EXPORT ~on_exit();

            PIKA_EXPORT on_exit(on_exit const& rhs) = delete;
            PIKA_EXPORT on_exit& operator=(on_exit const& rhs) = delete;

            PIKA_EXPORT on_exit(on_exit&& rhs) noexcept;
            PIKA_EXPORT on_exit& operator=(on_exit&& rhs) noexcept;

            pika::lcos::local::latch* latch_;
        };

    public:
        // Spawns a task to compute f() and returns immediately.
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_executor_any_v<std::decay_t<Executor>>
            )>
        // clang-format on
        void run(Executor&& exec, F&& f, Ts&&... ts)
        {
            // make sure exceptions don't leave the latch in the wrong state
            on_exit l(*this);

            pika::parallel::execution::post(PIKA_FORWARD(Executor, exec),
                [this, l = PIKA_MOVE(l), f = PIKA_FORWARD(F, f),
                    t = pika::make_tuple(PIKA_FORWARD(Ts, ts)...)]() mutable {
                    // latch needs to be released before the lambda exits
                    on_exit _(PIKA_MOVE(l));
                    std::exception_ptr p;
                    try
                    {
                        pika::util::invoke_fused(PIKA_MOVE(f), PIKA_MOVE(t));
                        return;
                    }
                    catch (...)
                    {
                        p = std::current_exception();
                    }

                    // The exception is set outside the catch block since
                    // set_exception may yield. Ending the catch block on a
                    // different worker thread than where it was started may
                    // lead to segfaults.
                    add_exception(PIKA_MOVE(p));
                });
        }

        // clang-format off
        template <typename F, typename... Ts,
            PIKA_CONCEPT_REQUIRES_(
                !pika::traits::is_executor_any_v<std::decay_t<F>>
            )>
        // clang-format on
        void run(F&& f, Ts&&... ts)
        {
            run(execution::parallel_executor{}, PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...);
        }

        // Waits for all tasks in the group to complete.
        PIKA_EXPORT void wait();

        // Add an exception to this task_group
        PIKA_EXPORT void add_exception(std::exception_ptr p);

    private:
        friend class serialization::access;

        PIKA_EXPORT void serialize(
            serialization::input_archive&, unsigned const);
        PIKA_EXPORT void serialize(
            serialization::output_archive&, unsigned const);

    private:
        using shared_state_type = lcos::detail::future_data<void>;

        pika::lcos::local::latch latch_;
        pika::intrusive_ptr<shared_state_type> state_;
        pika::exception_list errors_;
        std::atomic<bool> has_arrived_;
    };
}}}    // namespace pika::execution::experimental
