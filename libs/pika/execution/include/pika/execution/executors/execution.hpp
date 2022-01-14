//  Copyright (c) 2017-2021 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/execution.hpp

#pragma once

#include <pika/local/config.hpp>
// Necessary to avoid circular include
#include <pika/execution_base/execution.hpp>

#include <pika/assert.hpp>
#include <pika/async_combinators/wait_all.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/execution/executors/fused_bulk_execute.hpp>
#include <pika/execution/traits/executor_traits.hpp>
#include <pika/execution/traits/future_then_result_exec.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/functional/bind_back.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/modules/errors.hpp>
#include <pika/pack_traversal/unwrap.hpp>
#include <pika/type_support/detail/wrap_int.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace execution {
    /// \cond NOINTERNAL

    // customization point for OneWayExecutor interface
    // execute()
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename... Ts>
        struct sync_execute_not_callable;

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the sync_execute() customization point
        template <typename Executor, typename F, typename... Ts>
        PIKA_FORCEINLINE auto sync_execute_dispatch(
            pika::traits::detail::wrap_int, Executor&& /* exec */, F&& /* f */,
            Ts&&... /* ts */) -> sync_execute_not_callable<Executor, F, Ts...>
        {
            return sync_execute_not_callable<Executor, F, Ts...>{};
        }

        template <typename OneWayExecutor, typename F, typename... Ts>
        PIKA_FORCEINLINE auto sync_execute_dispatch(int, OneWayExecutor&& exec,
            F&& f, Ts&&... ts) -> decltype(exec.sync_execute(PIKA_FORWARD(F, f),
            PIKA_FORWARD(Ts, ts)...))
        {
            return exec.sync_execute(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        // emulate async_execute() on OneWayExecutors
        template <typename Executor>
        struct async_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_one_way_executor_v<Executor> &&
                !pika::traits::is_two_way_executor_v<Executor>>>
        {
            template <typename OneWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(
                std::false_type, OneWayExecutor&& exec, F&& f, Ts&&... ts)
                -> pika::future<decltype(
                    sync_execute_dispatch(0, PIKA_FORWARD(OneWayExecutor, exec),
                        PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))>
            {
                return pika::make_ready_future(
                    sync_execute_dispatch(0, PIKA_FORWARD(OneWayExecutor, exec),
                        PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...));
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static pika::future<void> call_impl(
                std::true_type, OneWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                sync_execute_dispatch(0, PIKA_FORWARD(OneWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
                return pika::make_ready_future();
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call(OneWayExecutor&& exec, F&& f,
                Ts&&... ts) -> pika::future<decltype(sync_execute_dispatch(0,
                PIKA_FORWARD(OneWayExecutor, exec), PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))>
            {
                using is_void = std::is_void<decltype(
                    sync_execute_dispatch(0, PIKA_FORWARD(OneWayExecutor, exec),
                        PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))>;

                return call_impl(
                    is_void(), exec, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<OneWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        // emulate sync_execute() on OneWayExecutors
        template <typename Executor>
        struct sync_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_one_way_executor_v<Executor> &&
                !pika::traits::is_two_way_executor_v<Executor>>>
        {
            template <typename OneWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call(OneWayExecutor&& exec, F&& f,
                Ts&&... ts) -> decltype(sync_execute_dispatch(0,
                PIKA_FORWARD(OneWayExecutor, exec), PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))
            {
                return sync_execute_dispatch(0,
                    PIKA_FORWARD(OneWayExecutor, exec), PIKA_FORWARD(F, f),
                    PIKA_FORWARD(Ts, ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<OneWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        // emulate then_execute() on OneWayExecutors
        template <typename Executor>
        struct then_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_one_way_executor_v<Executor> &&
                !pika::traits::is_two_way_executor_v<Executor>>>
        {
            template <typename OneWayExecutor, typename F, typename Future,
                typename... Ts>
            PIKA_FORCEINLINE static pika::future<
                pika::util::detail::invoke_deferred_result_t<F, Future, Ts...>>
            call(OneWayExecutor&& exec, F&& f, Future&& predecessor, Ts&&... ts)
            {
                using result_type =
                    pika::util::detail::invoke_deferred_result_t<F, Future,
                        Ts...>;

                auto func = pika::util::one_shot(pika::util::bind_back(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...));

                pika::traits::detail::shared_state_ptr_t<result_type> p =
                    lcos::detail::make_continuation_exec<result_type>(
                        PIKA_FORWARD(Future, predecessor),
                        PIKA_FORWARD(OneWayExecutor, exec), PIKA_MOVE(func));

                return pika::traits::future_access<
                    pika::future<result_type>>::create(PIKA_MOVE(p));
            }

            template <typename OneWayExecutor, typename F, typename Future,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<OneWayExecutor>(), std::declval<F>(),
                        std::declval<Future>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // emulate post() on OneWayExecutors
        template <typename Executor>
        struct post_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_one_way_executor_v<Executor> &&
                !pika::traits::is_two_way_executor_v<Executor> &&
                !pika::traits::is_never_blocking_one_way_executor_v<Executor>>>
        {
            template <typename OneWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static void call_impl(pika::traits::detail::wrap_int,
                OneWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                // execute synchronously
                sync_execute_dispatch(0, PIKA_FORWARD(OneWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            // dispatch to V1 executors
            template <typename OneWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(int, OneWayExecutor&& exec,
                F&& f, Ts&&... ts) -> decltype(exec.post(PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))
            {
                // use post, if exposed
                return exec.post(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call(OneWayExecutor&& exec, F&& f,
                Ts&&... ts) -> decltype(call_impl(0, exec, PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))
            {
                // simply discard the returned future
                return call_impl(0, PIKA_FORWARD(OneWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<OneWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // customization points for TwoWayExecutor interface
    // async_execute(), sync_execute(), then_execute()
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename... Ts>
        struct async_execute_not_callable;

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the async_execute() customization point
        template <typename Executor, typename F, typename... Ts>
        PIKA_FORCEINLINE auto async_execute_dispatch(
            pika::traits::detail::wrap_int, Executor&& /* exec */, F&& /* f */,
            Ts&&... /* ts */) -> async_execute_not_callable<Executor, F, Ts...>
        {
            return async_execute_not_callable<Executor, F, Ts...>{};
        }

        template <typename TwoWayExecutor, typename F, typename... Ts>
        PIKA_FORCEINLINE auto async_execute_dispatch(int, TwoWayExecutor&& exec,
            F&& f, Ts&&... ts) -> decltype(exec.async_execute(PIKA_FORWARD(F, f),
            PIKA_FORWARD(Ts, ts)...))
        {
            return exec.async_execute(
                PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        template <typename Executor>
        struct async_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_two_way_executor_v<Executor>>>
        {
            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call(TwoWayExecutor&& exec, F&& f,
                Ts&&... ts) -> decltype(async_execute_dispatch(0,
                PIKA_FORWARD(TwoWayExecutor, exec), PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))
            {
                return async_execute_dispatch(0,
                    PIKA_FORWARD(TwoWayExecutor, exec), PIKA_FORWARD(F, f),
                    PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<TwoWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the sync_execute() customization point
        template <typename Executor>
        struct sync_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_two_way_executor_v<Executor>>>
        {
            // fall-back: emulate sync_execute using async_execute
            template <typename TwoWayExecutor, typename F, typename... Ts>
            static auto call_impl(std::false_type, TwoWayExecutor&& exec, F&& f,
                Ts&&... ts) -> pika::util::invoke_result_t<F, Ts...>
            {
                try
                {
                    using result_type =
                        pika::util::detail::invoke_deferred_result_t<F, Ts...>;

                    // use async execution, wait for result, propagate exceptions
                    return async_execute_dispatch(0,
                        PIKA_FORWARD(TwoWayExecutor, exec),
                        [&]() -> result_type {
                            return PIKA_INVOKE(
                                PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
                        })
                        .get();
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    // note: constructor doesn't lock/suspend
                    throw pika::exception_list(std::current_exception());
                }
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static void call_impl(
                std::true_type, TwoWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                async_execute_dispatch(0, PIKA_FORWARD(TwoWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...)
                    .get();
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(pika::traits::detail::wrap_int,
                TwoWayExecutor&& exec, F&& f, Ts&&... ts)
                -> pika::util::invoke_result_t<F, Ts...>
            {
                using is_void = typename std::is_void<pika::util::detail::
                        invoke_deferred_result_t<F, Ts...>>::type;

                return call_impl(is_void(), PIKA_FORWARD(TwoWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(
                int, TwoWayExecutor&& exec, F&& f, Ts&&... ts)
                -> decltype(exec.sync_execute(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))
            {
                return exec.sync_execute(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call(
                TwoWayExecutor&& exec, F&& f, Ts&&... ts)
                -> decltype(call_impl(0, PIKA_FORWARD(TwoWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(TwoWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<TwoWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // then_execute()

        template <typename Executor>
        struct then_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_two_way_executor_v<Executor>>>
        {
            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            static pika::future<
                pika::util::detail::invoke_deferred_result_t<F, Future, Ts...>>
            call_impl(pika::traits::detail::wrap_int, TwoWayExecutor&& exec,
                F&& f, Future&& predecessor, Ts&&... ts)
            {
                using result_type =
                    pika::util::detail::invoke_deferred_result_t<F, Future,
                        Ts...>;

                auto func = pika::util::one_shot(pika::util::bind_back(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...));

                pika::traits::detail::shared_state_ptr_t<result_type> p =
                    lcos::detail::make_continuation_exec<result_type>(
                        PIKA_FORWARD(Future, predecessor),
                        PIKA_FORWARD(TwoWayExecutor, exec), PIKA_MOVE(func));

                return pika::traits::future_access<
                    pika::future<result_type>>::create(PIKA_MOVE(p));
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(int, TwoWayExecutor&& exec,
                F&& f, Future&& predecessor, Ts&&... ts)
                -> decltype(exec.then_execute(PIKA_FORWARD(F, f),
                    PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...))
            {
                return exec.then_execute(PIKA_FORWARD(F, f),
                    PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            PIKA_FORCEINLINE static auto call(TwoWayExecutor&& exec, F&& f,
                Future&& predecessor, Ts&&... ts) -> decltype(call_impl(0,
                PIKA_FORWARD(TwoWayExecutor, exec), PIKA_FORWARD(F, f),
                PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(TwoWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Future, predecessor),
                    PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<TwoWayExecutor>(), std::declval<F>(),
                        std::declval<Future>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // emulate post() on TwoWayExecutors
        template <typename Executor>
        struct post_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_two_way_executor_v<Executor> &&
                !pika::traits::is_never_blocking_one_way_executor_v<Executor>>>
        {
            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static void call_impl(pika::traits::detail::wrap_int,
                TwoWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                // simply discard the returned future
                exec.async_execute(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            // dispatch to V1 executors
            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(int, TwoWayExecutor&& exec,
                F&& f, Ts&&... ts) -> decltype(exec.post(PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))
            {
                // use post, if exposed
                return exec.post(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            PIKA_FORCEINLINE static auto call(
                TwoWayExecutor&& exec, F&& f, Ts&&... ts)
                -> decltype(call_impl(0, PIKA_FORWARD(TwoWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(TwoWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<TwoWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // post()
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // default implementation of the post() customization point

        template <typename Executor, typename F, typename... Ts>
        struct post_not_callable;

        template <typename Executor, typename F, typename... Ts>
        PIKA_FORCEINLINE auto post_dispatch(pika::traits::detail::wrap_int,
            Executor&& /* exec */, F&& /* f */, Ts&&... /* ts */)
            -> post_not_callable<Executor, F, Ts...>
        {
            return post_not_callable<Executor, F, Ts...>{};
        }

        // default implementation of the post() customization point
        template <typename NonBlockingOneWayExecutor, typename F,
            typename... Ts>
        PIKA_FORCEINLINE auto post_dispatch(
            int, NonBlockingOneWayExecutor&& exec, F&& f, Ts&&... ts)
            -> decltype(exec.post(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))
        {
            return exec.post(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
        }

        template <typename Executor>
        struct post_fn_helper<Executor,
            std::enable_if_t<
                pika::traits::is_never_blocking_one_way_executor_v<Executor>>>
        {
            template <typename NonBlockingOneWayExecutor, typename F,
                typename... Ts>
            PIKA_FORCEINLINE static auto call(NonBlockingOneWayExecutor&& exec,
                F&& f, Ts&&... ts) -> decltype(post_dispatch(0,
                PIKA_FORWARD(NonBlockingOneWayExecutor, exec), PIKA_FORWARD(F, f),
                PIKA_FORWARD(Ts, ts)...))
            {
                return post_dispatch(0,
                    PIKA_FORWARD(NonBlockingOneWayExecutor, exec),
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename NonBlockingOneWayExecutor, typename F,
                typename... Ts>
            struct result
            {
                using type =
                    decltype(call(std::declval<NonBlockingOneWayExecutor>(),
                        std::declval<F>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // customization points for BulkTwoWayExecutor interface
    // bulk_async_execute(), bulk_sync_execute(), bulk_then_execute()

    /// \cond NOINTERNAL
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // bulk_async_execute()

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the bulk_async_execute() customization point

        template <typename Executor, typename F, typename Shape, typename... Ts>
        struct bulk_async_execute_not_callable;

        template <typename Executor, typename F, typename Shape, typename... Ts>
        auto bulk_async_execute_dispatch(pika::traits::detail::wrap_int,
            Executor&& /* exec */, F&& /* f */, Shape const& /* shape */,
            Ts&&... /* ts */)
            -> bulk_async_execute_not_callable<Executor, F, Shape, Ts...>
        {
            return bulk_async_execute_not_callable<Executor, F, Shape, Ts...>{};
        }

        template <typename BulkTwoWayExecutor, typename F, typename Shape,
            typename... Ts>
        PIKA_FORCEINLINE auto bulk_async_execute_dispatch(int,
            BulkTwoWayExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
            -> decltype(exec.bulk_async_execute(
                PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
        {
            return exec.bulk_async_execute(
                PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
        }

        template <typename F, typename Shape, typename... Ts>
        struct bulk_function_result
        {
            using value_type =
                typename pika::traits::range_traits<Shape>::value_type;
            using type = pika::util::detail::invoke_deferred_result_t<F,
                value_type, Ts...>;
        };

        template <typename F, typename Shape, typename... Ts>
        using bulk_function_result_t =
            typename bulk_function_result<F, Shape, Ts...>::type;

        template <typename Executor>
        struct bulk_async_execute_fn_helper<Executor,
            std::enable_if_t<(pika::traits::is_one_way_executor_v<Executor> ||
                pika::traits::is_two_way_executor_v<Executor>) &&!pika::traits::
                    is_bulk_two_way_executor_v<Executor>>>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(pika::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> std::vector<pika::traits::executor_future_t<Executor,
                    bulk_function_result_t<F, Shape, Ts...>, Ts...>>
            {
                std::vector<pika::traits::executor_future_t<Executor,
                    bulk_function_result_t<F, Shape, Ts...>, Ts...>>
                    results;
                results.reserve(pika::util::size(shape));

                for (auto const& elem : shape)
                {
                    results.push_back(
                        execution::async_execute(exec, f, elem, ts...));
                }

                return results;
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(
                int, BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(exec.bulk_async_execute(
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
            {
                return exec.bulk_async_execute(
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        template <typename Executor>
        struct bulk_async_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_bulk_two_way_executor_v<Executor>>>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(bulk_async_execute_dispatch(0,
                    PIKA_FORWARD(BulkExecutor, exec), PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Ts, ts)...))
            {
                return bulk_async_execute_dispatch(0,
                    PIKA_FORWARD(BulkExecutor, exec), PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // bulk_sync_execute()

        // default implementation of the bulk_sync_execute() customization point
        template <typename Executor, typename F, typename Shape, typename... Ts>
        struct bulk_sync_execute_not_callable;

        template <typename Executor, typename F, typename Shape, typename... Ts>
        auto bulk_sync_execute_dispatch(pika::traits::detail::wrap_int,
            Executor&& /* exec */, F&& /* f */, Shape const& /* shape */,
            Ts&&... /* ts */)
            -> bulk_sync_execute_not_callable<Executor, F, Shape, Ts...>
        {
            return bulk_sync_execute_not_callable<Executor, F, Shape, Ts...>{};
        }

        template <typename BulkTwoWayExecutor, typename F, typename Shape,
            typename... Ts>
        PIKA_FORCEINLINE auto bulk_sync_execute_dispatch(int,
            BulkTwoWayExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
            -> decltype(exec.bulk_sync_execute(
                PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
        {
            return exec.bulk_sync_execute(
                PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, bool IsVoid, typename... Ts>
        struct bulk_execute_result_impl;

        template <typename F, typename Shape, typename... Ts>
        struct bulk_execute_result_impl<F, Shape, false, Ts...>
        {
            using type = std::vector<bulk_function_result_t<F, Shape, Ts...>>;
        };

        template <typename F, typename Shape, typename... Ts>
        struct bulk_execute_result_impl<F, Shape, true, Ts...>
        {
            using type = void;
        };

        template <typename F, typename Shape, bool IsVoid, typename... Ts>
        using bulk_execute_result_impl_t =
            typename bulk_execute_result_impl<F, Shape, IsVoid, Ts...>::type;

        template <typename F, typename Shape, typename... Ts>
        struct bulk_execute_result
          : bulk_execute_result_impl<F, Shape,
                std::is_void<bulk_function_result_t<F, Shape, Ts...>>::value,
                Ts...>
        {
        };

        template <typename F, typename Shape, typename... Ts>
        using bulk_execute_result_t =
            typename bulk_execute_result<F, Shape, Ts...>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct bulk_sync_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_one_way_executor_v<Executor> &&
                !pika::traits::is_two_way_executor_v<Executor> &&
                !pika::traits::is_bulk_one_way_executor_v<Executor>>>
        {
            // returns void if F returns void
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(std::false_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts)
                -> bulk_execute_result_impl_t<F, Shape, false, Ts...>
            {
                try
                {
                    bulk_execute_result_impl_t<F, Shape, false, Ts...> results;
                    results.reserve(pika::util::size(shape));

                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::sync_execute(exec, f, elem, ts...));
                    }
                    return results;
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    // note: constructor doesn't lock/suspend
                    throw pika::exception_list(std::current_exception());
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static void call_impl(std::true_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts)
            {
                try
                {
                    for (auto const& elem : shape)
                    {
                        execution::sync_execute(exec, f, elem, ts...);
                    }
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    // note: constructor doesn't lock/suspend
                    throw pika::exception_list(std::current_exception());
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(pika::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> bulk_execute_result_t<F, Shape, Ts...>
            {
                using is_void = typename std::is_void<
                    bulk_function_result_t<F, Shape, Ts...>>::type;

                return call_impl(is_void(), PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(
                int, BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(exec.bulk_sync_execute(
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
            {
                return exec.bulk_sync_execute(
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        template <typename Executor>
        struct bulk_sync_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_two_way_executor_v<Executor> &&
                !pika::traits::is_bulk_one_way_executor_v<Executor>>>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(std::false_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts)
                -> bulk_execute_result_t<F, Shape, Ts...>
            {
                using result_type =
                    std::vector<pika::traits::executor_future_t<Executor,
                        bulk_function_result_t<F, Shape, Ts...>>>;

                try
                {
                    result_type results;
                    results.reserve(pika::util::size(shape));
                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::async_execute(exec, f, elem, ts...));
                    }
                    return pika::unwrap(results);
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    // note: constructor doesn't lock/suspend
                    throw pika::exception_list(std::current_exception());
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static void call_impl(std::true_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts)
            {
                using result_type =
                    std::vector<pika::traits::executor_future_t<Executor,
                        bulk_function_result_t<F, Shape, Ts...>>>;

                result_type results;
                try
                {
                    results.reserve(pika::util::size(shape));

                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::async_execute(exec, f, elem, ts...));
                    }

                    pika::wait_all_nothrow(results);
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    // note: constructor doesn't lock/suspend
                    throw pika::exception_list(std::current_exception());
                }

                // handle exceptions
                pika::exception_list exceptions;
                for (auto& f : results)
                {
                    if (f.has_exception())
                    {
                        exceptions.add(f.get_exception_ptr());
                    }
                }

                if (exceptions.size() != 0)
                {
                    throw exceptions;
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(pika::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> bulk_execute_result_t<F, Shape, Ts...>
            {
                using is_void = typename std::is_void<
                    bulk_function_result_t<F, Shape, Ts...>>::type;

                return call_impl(is_void(), PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(
                int, BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(exec.bulk_sync_execute(
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
            {
                return exec.bulk_sync_execute(
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        template <typename Executor>
        struct bulk_sync_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_bulk_one_way_executor_v<Executor>>>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            PIKA_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(bulk_sync_execute_dispatch(0,
                    PIKA_FORWARD(BulkExecutor, exec), PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Ts, ts)...))
            {
                return bulk_sync_execute_dispatch(0,
                    PIKA_FORWARD(BulkExecutor, exec), PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail
    /// \endcond

    /// \cond NOINTERNAL
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // bulk_then_execute()

        template <typename Executor>
        struct bulk_then_execute_fn_helper<Executor,
            std::enable_if_t<
                !pika::traits::is_bulk_two_way_executor_v<Executor>>>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            static auto call_impl(std::false_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> pika::future<
                    bulk_then_execute_result_t<F, Shape, Future, Ts...>>
            {
                using result_type =
                    bulk_then_execute_result_t<F, Shape, Future, Ts...>;

                using shared_state_type =
                    pika::traits::detail::shared_state_ptr_t<result_type>;

                auto func = make_fused_bulk_sync_execute_helper<result_type>(
                    exec, PIKA_FORWARD(F, f), shape,
                    pika::make_tuple(PIKA_FORWARD(Ts, ts)...));

                shared_state_type p =
                    lcos::detail::make_continuation_exec<result_type>(
                        PIKA_FORWARD(Future, predecessor),
                        PIKA_FORWARD(BulkExecutor, exec), PIKA_MOVE(func));

                return pika::traits::future_access<
                    pika::future<result_type>>::create(PIKA_MOVE(p));
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            static pika::future<void> call_impl(std::true_type,
                BulkExecutor&& exec, F&& f, Shape const& shape,
                Future&& predecessor, Ts&&... ts)
            {
                auto func = make_fused_bulk_sync_execute_helper<void>(exec,
                    PIKA_FORWARD(F, f), shape,
                    pika::make_tuple(PIKA_FORWARD(Ts, ts)...));

                pika::traits::detail::shared_state_ptr_t<void> p =
                    lcos::detail::make_continuation_exec<void>(
                        PIKA_FORWARD(Future, predecessor),
                        PIKA_FORWARD(BulkExecutor, exec), PIKA_MOVE(func));

                return pika::traits::future_access<pika::future<void>>::create(
                    PIKA_MOVE(p));
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(pika::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape,
                Future&& predecessor, Ts&&... ts)
                -> pika::future<
                    bulk_then_execute_result_t<F, Shape, Future, Ts...>>
            {
                using is_void = typename std::is_void<
                    then_bulk_function_result_t<F, Shape, Future, Ts...>>::type;

                return bulk_then_execute_fn_helper::call_impl(is_void(),
                    PIKA_FORWARD(BulkExecutor, exec), PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(int, BulkExecutor&& exec,
                F&& f, Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(exec.bulk_then_execute(PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...))
            {
                return exec.bulk_then_execute(PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            PIKA_FORCEINLINE static auto call(BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape,
                    pika::make_shared_future(PIKA_FORWARD(Future, predecessor)),
                    PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape,
                    pika::make_shared_future(PIKA_FORWARD(Future, predecessor)),
                    PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<BulkExecutor>(),
                    std::declval<F>(), std::declval<Shape const&>(),
                    std::declval<Future>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct bulk_then_execute_fn_helper<Executor,
            std::enable_if_t<pika::traits::is_bulk_two_way_executor_v<Executor>>>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            static auto call_impl(
                pika::traits::detail::wrap_int, BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor,
                Ts&&...
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
                ts
#endif
                ) -> pika::traits::executor_future_t<Executor,
                bulk_then_execute_result_t<F, Shape, Future, Ts...>>
            {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
                PIKA_UNUSED(exec);
                PIKA_UNUSED(f);
                PIKA_UNUSED(shape);
                PIKA_UNUSED(predecessor);
                PIKA_ASSERT(false);
                return pika::traits::executor_future_t<Executor,
                    bulk_then_execute_result_t<F, Shape, Future, Ts...>>{};
#else
                // result_of_t<F(Shape::value_type, Future)>
                using func_result_type =
                    then_bulk_function_result_t<F, Shape, Future, Ts...>;

                // std::vector<future<func_result_type>>
                using result_type =
                    std::vector<pika::traits::executor_future_t<Executor,
                        func_result_type, Ts...>>;

                auto func = make_fused_bulk_async_execute_helper<result_type>(
                    exec, PIKA_FORWARD(F, f), shape,
                    pika::make_tuple(PIKA_FORWARD(Ts, ts)...));

                // void or std::vector<func_result_type>
                using vector_result_type =
                    bulk_then_execute_result_t<F, Shape, Future, Ts...>;

                // future<vector_result_type>
                using result_future_type =
                    pika::traits::executor_future_t<Executor,
                        vector_result_type>;

                using shared_state_type =
                    pika::traits::detail::shared_state_ptr_t<vector_result_type>;

                using future_type = std::decay_t<Future>;

                shared_state_type p =
                    lcos::detail::make_continuation_exec<vector_result_type>(
                        PIKA_FORWARD(Future, predecessor),
                        PIKA_FORWARD(BulkExecutor, exec),
                        [func = PIKA_MOVE(func)](
                            future_type&& predecessor) mutable
                        -> vector_result_type {
                            // use unwrap directly (instead of lazily) to avoid
                            // having to pull in dataflow
                            return pika::unwrap(func(PIKA_MOVE(predecessor)));
                        });

                return pika::traits::future_access<result_future_type>::create(
                    PIKA_MOVE(p));
#endif
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            PIKA_FORCEINLINE static auto call_impl(int, BulkExecutor&& exec,
                F&& f, Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(exec.bulk_then_execute(PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...))
            {
                return exec.bulk_then_execute(PIKA_FORWARD(F, f), shape,
                    PIKA_FORWARD(Future, predecessor), PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            PIKA_FORCEINLINE static auto call(BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape,
                    pika::make_shared_future(PIKA_FORWARD(Future, predecessor)),
                    PIKA_FORWARD(Ts, ts)...))
            {
                return call_impl(0, PIKA_FORWARD(BulkExecutor, exec),
                    PIKA_FORWARD(F, f), shape,
                    pika::make_shared_future(PIKA_FORWARD(Future, predecessor)),
                    PIKA_FORWARD(Ts, ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<BulkExecutor>(),
                    std::declval<F>(), std::declval<Shape const&>(),
                    std::declval<Future>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail
    /// \endcond
}}}    // namespace pika::parallel::execution
