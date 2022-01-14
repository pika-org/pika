//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/scheduler_executor.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/execution/algorithms/bulk.hpp>
#include <pika/execution/algorithms/keep_future.hpp>
#include <pika/execution/algorithms/make_future.hpp>
#include <pika/execution/algorithms/start_detached.hpp>
#include <pika/execution/algorithms/sync_wait.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/execution_base/execution.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/functional/bind_back.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/functional/invoke_fused.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace execution { namespace experimental {

    namespace detail {
#if defined(PIKA_HAVE_CXX20_PERFECT_PACK_CAPTURE)
        template <typename F, typename... Ts>
        auto captured_args_then(F&& f, Ts&&... ts)
        {
            return [f = PIKA_FORWARD(F, f), ... ts = PIKA_FORWARD(Ts, ts)](
                       auto i, auto&& predecessor, auto& v) mutable {
                v[i] = PIKA_INVOKE(PIKA_FORWARD(F, f), i,
                    PIKA_FORWARD(decltype(predecessor), predecessor),
                    PIKA_FORWARD(Ts, ts)...);
            };
        }
#else
        template <typename F, typename... Ts>
        auto captured_args_then(F&& f, Ts&&... ts)
        {
            return [f = PIKA_FORWARD(F, f),
                       t = pika::make_tuple(PIKA_FORWARD(Ts, ts)...)](
                       auto i, auto&& predecessor, auto& v) mutable {
                v[i] = pika::util::invoke_fused(
                    pika::util::bind_front(PIKA_FORWARD(F, f), i,
                        PIKA_FORWARD(decltype(predecessor), predecessor)),
                    PIKA_MOVE(t));
            };
        }
#endif
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // A scheduler_executor wraps any P2300 scheduler and implements the
    // executor functionalities for those.
    template <typename BaseScheduler>
    struct scheduler_executor
    {
        static_assert(pika::execution::experimental::is_scheduler_v<
                          std::decay_t<BaseScheduler>>,
            "scheduler_executor requires a scheduler");

        constexpr scheduler_executor() = default;

        template <typename Scheduler,
            typename Enable = std::enable_if_t<
                pika::execution::experimental::is_scheduler_v<Scheduler>>>
        constexpr explicit scheduler_executor(Scheduler&& sched)
          : sched_(PIKA_FORWARD(Scheduler, sched))
        {
        }

        constexpr scheduler_executor(scheduler_executor&&) = default;
        constexpr scheduler_executor& operator=(scheduler_executor&&) = default;
        constexpr scheduler_executor(scheduler_executor const&) = default;
        constexpr scheduler_executor& operator=(
            scheduler_executor const&) = default;

        /// \cond NOINTERNAL
        constexpr bool operator==(scheduler_executor const& rhs) const noexcept
        {
            return sched_ == rhs.sched_;
        }

        constexpr bool operator!=(scheduler_executor const& rhs) const noexcept
        {
            return sched_ != rhs.sched_;
        }

        constexpr auto const& context() const noexcept
        {
            return *this;
        }

        template <typename Enable =
                      std::enable_if_t<pika::is_invocable_v<with_priority_t,
                          BaseScheduler, pika::threads::thread_priority>>>
        friend scheduler_executor tag_invoke(
            pika::execution::experimental::with_priority_t,
            scheduler_executor const& exec,
            pika::threads::thread_priority priority)
        {
            return scheduler_executor(with_priority(exec.sched_, priority));
        }

        template <typename Enable = std::enable_if_t<
                      pika::is_invocable_v<get_priority_t, BaseScheduler>>>
        friend pika::threads::thread_priority tag_invoke(
            pika::execution::experimental::get_priority_t,
            scheduler_executor const& exec)
        {
            return get_priority(exec.sched_);
        }

        template <typename Enable =
                      std::enable_if_t<pika::is_invocable_v<with_stacksize_t,
                          BaseScheduler, pika::threads::thread_stacksize>>>
        friend scheduler_executor tag_invoke(
            pika::execution::experimental::with_stacksize_t,
            scheduler_executor const& exec,
            pika::threads::thread_stacksize stacksize)
        {
            return scheduler_executor(with_stacksize(exec.sched_, stacksize));
        }

        template <typename Enable = std::enable_if_t<
                      pika::is_invocable_v<get_stacksize_t, BaseScheduler>>>
        friend pika::threads::thread_stacksize tag_invoke(
            pika::execution::experimental::get_stacksize_t,
            scheduler_executor const& exec)
        {
            return get_stacksize(exec.sched_);
        }

        template <
            typename Enable = std::enable_if_t<pika::is_invocable_v<with_hint_t,
                BaseScheduler, pika::threads::thread_schedule_hint>>>
        friend scheduler_executor tag_invoke(
            pika::execution::experimental::with_hint_t,
            scheduler_executor const& exec,
            pika::threads::thread_schedule_hint hint)
        {
            return scheduler_executor(with_hint(exec.sched_, hint));
        }

        template <typename Enable = std::enable_if_t<
                      pika::is_invocable_v<get_hint_t, BaseScheduler>>>
        friend pika::threads::thread_schedule_hint tag_invoke(
            pika::execution::experimental::get_hint_t,
            scheduler_executor const& exec)
        {
            return get_hint(exec.sched_);
        }

        template <typename Enable = std::enable_if_t<pika::is_invocable_v<
                      with_annotation_t, BaseScheduler, char const*>>>
        friend scheduler_executor tag_invoke(
            pika::execution::experimental::with_annotation_t,
            scheduler_executor const& exec, char const* annotation)
        {
            return scheduler_executor(with_annotation(exec.sched_, annotation));
        }

        template <typename Enable = std::enable_if_t<pika::is_invocable_v<
                      with_annotation_t, BaseScheduler, std::string>>>
        friend scheduler_executor tag_invoke(
            pika::execution::experimental::with_annotation_t,
            scheduler_executor const& exec, std::string annotation)
        {
            return scheduler_executor(with_annotation(exec.sched_, annotation));
        }

        template <typename Enable = std::enable_if_t<
                      pika::is_invocable_v<get_annotation_t, BaseScheduler>>>
        friend char const* tag_invoke(
            pika::execution::experimental::get_annotation_t,
            scheduler_executor const& exec)
        {
            return get_annotation(exec.sched_);
        }

        // Associate the parallel_execution_tag executor tag type as a default
        // with this executor.
        using execution_category = parallel_execution_tag;

        // Associate the static_chunk_size executor parameters type as a default
        // with this executor.
        using executor_parameters_type = static_chunk_size;

        template <typename T, typename... Ts>
        using future_type = pika::future<T>;

        // NonBlockingOneWayExecutor interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            start_detached(then(schedule(sched_),
                pika::util::deferred_call(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...)));
        }

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) sync_execute(F&& f, Ts&&... ts)
        {
            return sync_wait(then(schedule(sched_),
                pika::util::deferred_call(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...)));
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return make_future(then(schedule(sched_),
                pika::util::deferred_call(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...)));
        }

        template <typename F, typename Future, typename... Ts>
        decltype(auto) then_execute(F&& f, Future&& predecessor, Ts&&... ts)
        {
            auto&& predecessor_transfer_sched =
                transfer(keep_future(PIKA_FORWARD(Future, predecessor)), sched_);

            return make_future(then(PIKA_MOVE(predecessor_transfer_sched),
                pika::util::bind_back(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...)));
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            using shape_element =
                typename pika::traits::range_traits<S>::value_type;
            using result_type = pika::util::detail::invoke_deferred_result_t<F,
                shape_element, Ts...>;

            if constexpr (std::is_void_v<result_type>)
            {
                std::vector<pika::future<void>> results;
                results.reserve(1);
                results.emplace_back(make_future(bulk(schedule(sched_), shape,
                    pika::util::bind_back(
                        PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))));
                return results;
            }
            else
            {
                using promise_vector_type =
                    std::vector<pika::lcos::local::promise<result_type>>;
                using result_vector_type =
                    std::vector<pika::future<result_type>>;

                using size_type = decltype(pika::util::size(shape));
                size_type const n = pika::util::size(shape);

                promise_vector_type promises(n);
                result_vector_type results;
                results.reserve(n);

                for (size_type i = 0; i < n; ++i)
                {
                    results.emplace_back(promises[i].get_future());
                }

                auto f_helper = [](size_type const i,
                                    promise_vector_type& promises, F& f,
                                    S const& shape, Ts&... ts) {
                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            auto it = pika::util::begin(shape);
                            std::advance(it, i);
                            promises[i].set_value(PIKA_INVOKE(f, *it, ts...));
                        },
                        [&](std::exception_ptr&& ep) {
                            promises[i].set_exception(PIKA_MOVE(ep));
                        });
                };

                start_detached(
                    bulk(transfer_just(sched_, PIKA_MOVE(promises),
                             PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...),
                        n, PIKA_MOVE(f_helper)));

                return results;
            }
        }

        template <typename F, typename S, typename... Ts>
        void bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
        {
            sync_wait(bulk(schedule(sched_), shape,
                pika::util::bind_back(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...)));
        }

        template <typename F, typename S, typename Future, typename... Ts>
        decltype(auto) bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            using result_type =
                parallel::execution::detail::then_bulk_function_result_t<F, S,
                    Future, Ts...>;

            if constexpr (std::is_void<result_type>::value)
            {
                // the overall return value is future<void>
                auto prereq =
                    when_all(keep_future(PIKA_FORWARD(Future, predecessor)));

                auto loop = bulk(transfer(PIKA_MOVE(prereq), sched_), shape,
                    pika::util::bind_back(
                        PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...));

                return make_future(PIKA_MOVE(loop));
            }
            else
            {
                // the overall return value is future<std::vector<result_type>>
                auto prereq =
                    when_all(keep_future(PIKA_FORWARD(Future, predecessor)),
                        just(std::vector<result_type>(pika::util::size(shape))));

                auto loop = bulk(transfer(PIKA_MOVE(prereq), sched_), shape,
                    detail::captured_args_then(
                        PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...));

                return make_future(then(
                    PIKA_MOVE(loop), [](auto&&, std::vector<result_type>&& v) {
                        return PIKA_MOVE(v);
                    }));
            }
        }

    private:
        BaseScheduler sched_;
        /// \endcond
    };

    template <typename BaseScheduler>
    explicit scheduler_executor(BaseScheduler&& sched)
        -> scheduler_executor<std::decay_t<BaseScheduler>>;
}}}    // namespace pika::execution::experimental

namespace pika { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <typename BaseScheduler>
    struct is_one_way_executor<
        pika::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_never_blocking_one_way_executor<
        pika::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_bulk_one_way_executor<
        pika::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_two_way_executor<
        pika::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_bulk_two_way_executor<
        pika::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace pika::parallel::execution
