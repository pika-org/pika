//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/async_base/scheduling_properties.hpp>
#include <pika/async_cuda/cuda_pool.hpp>
#include <pika/async_cuda_base/cuda_stream.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/execution/algorithms/execute.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>

#include <utility>

namespace pika::cuda::experimental {
    /// A scheduler for running work on a CUDA pool.
    ///
    /// Provides access to scheduling work on a CUDA context represented by a \ref cuda_pool. Models
    /// the [std::execution scheduler concept](https://eel.is/c++draft/exec.sched).
    ///
    /// Move and copy constructible. The scheduler has reference semantics with respect to the
    /// associated CUDA pool.
    ///
    /// Equality comparable.
    ///
    /// \note The recommended way to access streams and handles from the \ref cuda_pool is through
    /// the sender adaptors \ref then_with_stream, \ref then_with_cublas, and \ref
    /// then_with_cusolver.
    class cuda_scheduler
    {
    private:
        cuda_pool pool;
        pika::execution::thread_priority priority;

    public:
        /// \brief Constructs a new \ref cuda_scheduler using the given \ref cuda_pool.
        PIKA_EXPORT explicit cuda_scheduler(cuda_pool pool);
        cuda_scheduler(cuda_scheduler&&) = default;
        cuda_scheduler(cuda_scheduler const&) = default;
        cuda_scheduler& operator=(cuda_scheduler&&) = default;
        cuda_scheduler& operator=(cuda_scheduler const&) = default;
        ~cuda_scheduler(){};

        /// \brief Return the \ref cuda_pool associated with this scheduler.
        PIKA_EXPORT cuda_pool const& get_pool() const noexcept;

        /// \brief Return the next available CUDA stream from the pool.
        PIKA_EXPORT cuda_stream const& get_next_stream();

        /// \brief Return the next available cuBLAS handle from the pool.
        PIKA_EXPORT locked_cublas_handle get_cublas_handle(
            cuda_stream const& stream, cublasPointerMode_t pointer_mode);

        /// \brief Return the next available cuSOLVER handle from the pool.
        PIKA_EXPORT locked_cusolver_handle get_cusolver_handle(cuda_stream const& stream);

        /// \cond NOINTERNAL
        friend bool operator==(cuda_scheduler const& lhs, cuda_scheduler const& rhs)
        {
            return lhs.pool == rhs.pool;
        }

        friend bool operator!=(cuda_scheduler const& lhs, cuda_scheduler const& rhs)
        {
            return !(lhs == rhs);
        }

        friend cuda_scheduler tag_invoke(pika::execution::experimental::with_priority_t,
            cuda_scheduler const& scheduler, pika::execution::thread_priority priority)
        {
            auto sched_with_priority = scheduler;
            sched_with_priority.priority = priority;
            return sched_with_priority;
        }

        friend pika::execution::thread_priority tag_invoke(
            pika::execution::experimental::get_priority_t, cuda_scheduler const& scheduler)
        {
            return scheduler.priority;
        }
        /// \endcond
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

                template <typename Receiver_>
                operation_state(cuda_scheduler scheduler, Receiver_&& receiver)
                  : scheduler(std::move(scheduler))
                  , receiver(std::forward<Receiver_>(receiver))
                {
                }
                operation_state(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(
                    pika::execution::experimental::start_t, operation_state& os) noexcept
                {
                    // This currently only acts as an inline scheduler to signal
                    // downstream senders that they should use the
                    // cuda_scheduler.
                    pika::execution::experimental::set_value(std::move(os.receiver));
                }
            };

        public:
            PIKA_STDEXEC_SENDER_CONCEPT

            PIKA_EXPORT explicit cuda_scheduler_sender(cuda_scheduler scheduler);
            cuda_scheduler_sender(cuda_scheduler_sender&&) = default;
            cuda_scheduler_sender& operator=(cuda_scheduler_sender&&) = default;
            cuda_scheduler_sender(cuda_scheduler_sender const&) = delete;
            cuda_scheduler_sender& operator=(cuda_scheduler_sender const&) = delete;

#if defined(PIKA_HAVE_STDEXEC)
            using completion_signatures = pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t()>;
#else
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<>;

            static constexpr bool sends_done = false;
#endif

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(pika::execution::experimental::connect_t,
                cuda_scheduler_sender&& s, Receiver&& receiver)
            {
                return {std::move(s.scheduler), std::forward<Receiver>(receiver)};
            }

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(pika::execution::experimental::connect_t,
                cuda_scheduler_sender const& s, Receiver&& receiver)
            {
                return {s.scheduler, std::forward<Receiver>(receiver)};
            }

            struct env
            {
                cuda_scheduler scheduler;

                friend cuda_scheduler tag_invoke(
                    pika::execution::experimental::get_completion_scheduler_t<
                        pika::execution::experimental::set_value_t>,
                    env const& e) noexcept
                {
                    return e.scheduler;
                }
            };

            friend env tag_invoke(
                pika::execution::experimental::get_env_t, cuda_scheduler_sender const& s) noexcept
            {
                return {s.scheduler};
            }
        };
    }    // namespace detail

    /// Schedule subsequent work for execution on a CUDA device.
    inline auto tag_invoke(
        pika::execution::experimental::schedule_t, cuda_scheduler scheduler) noexcept
    {
        return detail::cuda_scheduler_sender{std::move(scheduler)};
    }
}    // namespace pika::cuda::experimental
