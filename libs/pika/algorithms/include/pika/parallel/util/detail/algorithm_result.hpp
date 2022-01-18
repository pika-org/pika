//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/executors/execution_policy_fwd.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/futures/future.hpp>
#include <pika/type_support/unused.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace parallel { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T>
    struct algorithm_result_impl
    {
        // The return type of the initiating function.
        using type = T;

        // Obtain initiating function's return type.
        static constexpr type get()
        {
            return T();
        }

        template <typename T_>
        static constexpr type get(T_&& t)
        {
            return PIKA_FORWARD(T_, t);
        }

        template <typename T_>
        static type get(pika::future<T_>&& t)
        {
            return t.get();
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void>
    {
        // The return type of the initiating function.
        using type = void;

        // Obtain initiating function's return type.
        static constexpr void get() noexcept {}

        static constexpr void get(pika::util::unused_type) noexcept {}

        static void get(pika::future<void>&& t)
        {
            t.get();
        }

        template <typename T>
        static void get(pika::future<T>&& t)
        {
            t.get();
        }
    };

    template <typename T>
    struct algorithm_result_impl<pika::execution::sequenced_task_policy, T>
    {
        // The return type of the initiating function.
        using type = pika::future<T>;

        // Obtain initiating function's return type.
        static type get(T&& t)
        {
            return pika::make_ready_future(PIKA_MOVE(t));
        }

        static type get(pika::future<T>&& t)
        {
            return PIKA_MOVE(t);
        }
    };

    template <>
    struct algorithm_result_impl<pika::execution::sequenced_task_policy, void>
    {
        // The return type of the initiating function.
        using type = pika::future<void>;

        // Obtain initiating function's return type.
        static type get()
        {
            return pika::make_ready_future();
        }

        static type get(pika::util::unused_type)
        {
            return pika::make_ready_future();
        }

        static type get(pika::future<void>&& t)
        {
            return PIKA_MOVE(t);
        }

        template <typename T>
        static type get(pika::future<T>&& t)
        {
            return pika::future<void>(PIKA_MOVE(t));
        }
    };

    template <typename T>
    struct algorithm_result_impl<pika::execution::parallel_task_policy, T>
    {
        // The return type of the initiating function.
        using type = pika::future<T>;

        // Obtain initiating function's return type.
        static type get(T&& t)
        {
            return pika::make_ready_future(PIKA_MOVE(t));
        }

        static type get(pika::future<T>&& t)
        {
            return PIKA_MOVE(t);
        }
    };

    template <>
    struct algorithm_result_impl<pika::execution::parallel_task_policy, void>
    {
        // The return type of the initiating function.
        using type = pika::future<void>;

        // Obtain initiating function's return type.
        static type get()
        {
            return pika::make_ready_future();
        }

        static type get(pika::util::unused_type)
        {
            return pika::make_ready_future();
        }

        static type get(pika::future<void>&& t)
        {
            return PIKA_MOVE(t);
        }

        template <typename T>
        static type get(pika::future<T>&& t)
        {
            return pika::future<void>(PIKA_MOVE(t));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        pika::execution::sequenced_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<pika::execution::sequenced_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        pika::execution::sequenced_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<pika::execution::sequenced_task_policy, void>
    {
    };

    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        pika::execution::parallel_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<pika::execution::parallel_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        pika::execution::parallel_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<pika::execution::parallel_task_policy, void>
    {
    };

#if defined(PIKA_HAVE_DATAPAR)
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct algorithm_result_impl<pika::execution::simd_task_policy, T>
      : algorithm_result_impl<pika::execution::sequenced_task_policy, T>
    {
    };

    template <>
    struct algorithm_result_impl<pika::execution::simd_task_policy, void>
      : algorithm_result_impl<pika::execution::sequenced_task_policy, void>
    {
    };

    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        pika::execution::simd_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<pika::execution::sequenced_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        pika::execution::simd_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<pika::execution::sequenced_task_policy, void>
    {
    };

    template <typename T>
    struct algorithm_result_impl<pika::execution::par_simd_task_policy, T>
      : algorithm_result_impl<pika::execution::parallel_task_policy, T>
    {
    };

    template <>
    struct algorithm_result_impl<pika::execution::par_simd_task_policy, void>
      : algorithm_result_impl<pika::execution::parallel_task_policy, void>
    {
    };

    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        pika::execution::par_simd_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<pika::execution::parallel_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        pika::execution::par_simd_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<pika::execution::parallel_task_policy, void>
    {
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T = void>
    struct algorithm_result
      : algorithm_result_impl<typename std::decay<ExPolicy>::type, T>
    {
        static_assert(!std::is_lvalue_reference<T>::value,
            "T shouldn't be a lvalue reference");
    };

    template <typename ExPolicy, typename T = void>
    using algorithm_result_t = typename algorithm_result<ExPolicy, T>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename U, typename Conv,
        PIKA_CONCEPT_REQUIRES_(pika::is_invocable_v<Conv, U>)>
    constexpr typename pika::util::invoke_result<Conv, U>::type
    convert_to_result(U&& val, Conv&& conv)
    {
        return PIKA_INVOKE(conv, val);
    }

    template <typename U, typename Conv,
        PIKA_CONCEPT_REQUIRES_(pika::is_invocable_v<Conv, U>)>
    pika::future<typename pika::util::invoke_result<Conv, U>::type>
    convert_to_result(pika::future<U>&& f, Conv&& conv)
    {
        using result_type = typename pika::util::invoke_result<Conv, U>::type;

        return pika::make_future<result_type>(
            PIKA_MOVE(f), PIKA_FORWARD(Conv, conv));
    }
}}}}    // namespace pika::parallel::util::detail
