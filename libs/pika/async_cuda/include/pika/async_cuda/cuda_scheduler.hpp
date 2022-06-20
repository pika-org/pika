//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/async_base/scheduling_properties.hpp>
#include <pika/async_cuda/cuda_exception.hpp>
#include <pika/async_cuda/cuda_pool.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/execution/algorithms/execute.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>

namespace pika::cuda::experimental {
    /// A scheduler for running work on a CUDA pool.
    ///
    /// Provides access to scheduling work on a CUDA context represented by a
    /// cuda_pool.
    class cuda_scheduler
    {
    private:
        cuda_pool pool;
        pika::execution::thread_priority priority;

    public:
        PIKA_EXPORT
        cuda_scheduler(cuda_pool pool);
        cuda_scheduler(cuda_scheduler&&) = default;
        cuda_scheduler(cuda_scheduler const&) = default;
        cuda_scheduler& operator=(cuda_scheduler&&) = default;
        cuda_scheduler& operator=(cuda_scheduler const&) = default;
        ~cuda_scheduler(){};

        PIKA_EXPORT cuda_pool const& get_pool() const noexcept;
        PIKA_EXPORT cuda_stream const& get_next_stream();

        /// \cond NOINTERNAL
        friend bool operator==(
            cuda_scheduler const& lhs, cuda_scheduler const& rhs)
        {
            return lhs.pool == rhs.pool;
        }

        friend bool operator!=(
            cuda_scheduler const& lhs, cuda_scheduler const& rhs)
        {
            return !(lhs == rhs);
        }
        /// \endcond

        friend cuda_scheduler tag_invoke(
            pika::execution::experimental::with_priority_t,
            cuda_scheduler const& scheduler,
            pika::execution::thread_priority priority)
        {
            auto sched_with_priority = scheduler;
            sched_with_priority.priority = priority;
            return sched_with_priority;
        }

        friend pika::execution::thread_priority tag_invoke(
            pika::execution::experimental::get_priority_t,
            cuda_scheduler const& scheduler)
        {
            return scheduler.priority;
        }
    };

    namespace detail {
        /// A sender that represents work starting on a CUDA device.
        class cuda_scheduler_sender
        {
        private:
            cuda_scheduler scheduler;

            template <typename Receiver>
            struct operation_state
            {
                cuda_scheduler scheduler;
                PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                operation_state(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(pika::execution::experimental::start_t,
                    operation_state& os) noexcept
                {
                    // This currently only acts as an inline scheduler to signal
                    // downstream senders that they should use the
                    // cuda_scheduler.
                    pika::execution::experimental::set_value(
                        PIKA_MOVE(os.receiver));
                }
            };

        public:
            PIKA_EXPORT explicit cuda_scheduler_sender(
                cuda_scheduler scheduler);
            cuda_scheduler_sender(cuda_scheduler_sender&&) = default;
            cuda_scheduler_sender& operator=(cuda_scheduler_sender&&) = default;
            cuda_scheduler_sender(cuda_scheduler_sender const&) = delete;
            cuda_scheduler_sender& operator=(
                cuda_scheduler_sender const&) = delete;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(
                pika::execution::experimental::connect_t,
                cuda_scheduler_sender&& s, Receiver&& receiver)
            {
                return {
                    PIKA_MOVE(s.scheduler), PIKA_FORWARD(Receiver, receiver)};
            }

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(
                pika::execution::experimental::connect_t,
                cuda_scheduler_sender const& s, Receiver&& receiver)
            {
                return {s.scheduler, PIKA_FORWARD(Receiver, receiver)};
            }

            friend cuda_scheduler tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<
                    pika::execution::experimental::set_value_t>,
                cuda_scheduler_sender const& s)
            {
                return s.scheduler;
            }
        };
    }    // namespace detail

    /// Schedule subsequent work for execution on a CUDA device.
    inline auto tag_invoke(pika::execution::experimental::schedule_t,
        cuda_scheduler scheduler) noexcept
    {
        return detail::cuda_scheduler_sender{PIKA_MOVE(scheduler)};
    }
}    // namespace pika::cuda::experimental
