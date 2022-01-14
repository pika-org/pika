//  Copyright (c) 2017-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/execution_base/execution.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_range.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { namespace execution { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename Future, typename... Ts>
    struct then_bulk_function_result
    {
        using value_type =
            typename pika::traits::range_traits<Shape>::value_type;
        using type = pika::util::detail::invoke_deferred_result_t<F, value_type,
            Future, Ts...>;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    using then_bulk_function_result_t =
        typename then_bulk_function_result<F, Shape, Future, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename Future, bool IsVoid,
        typename... Ts>
    struct bulk_then_execute_result_impl;

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result_impl<F, Shape, Future, false, Ts...>
    {
        using type =
            std::vector<then_bulk_function_result_t<F, Shape, Future, Ts...>>;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result_impl<F, Shape, Future, true, Ts...>
    {
        using type = void;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result
      : bulk_then_execute_result_impl<F, Shape, Future,
            std::is_void<
                then_bulk_function_result_t<F, Shape, Future, Ts...>>::value,
            Ts...>
    {
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    using bulk_then_execute_result_t =
        typename bulk_then_execute_result<F, Shape, Future, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename Shape, typename Future,
        std::size_t... Is, typename... Ts>
    PIKA_FORCEINLINE auto fused_bulk_sync_execute(Executor&& exec, F&& f,
        Shape const& shape, Future&& predecessor, pika::util::index_pack<Is...>,
        pika::tuple<Ts...> const& args)
        -> decltype(execution::bulk_sync_execute(PIKA_FORWARD(Executor, exec),
            PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Future, predecessor),
            pika::get<Is>(args)...))
    {
        return execution::bulk_sync_execute(PIKA_FORWARD(Executor, exec),
            PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Future, predecessor),
            pika::get<Is>(args)...);
    }

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    struct fused_bulk_sync_execute_helper;

    template <typename Result, typename Executor, typename F, typename Shape,
        typename... Ts>
    struct fused_bulk_sync_execute_helper<Result, Executor, F, Shape,
        pika::tuple<Ts...>>
    {
        Executor exec_;
        F f_;
        Shape shape_;
        pika::tuple<Ts...> args_;

        template <typename Future>
        Result operator()(Future&& predecessor)
        {
            return fused_bulk_sync_execute(exec_, f_, shape_,
                PIKA_FORWARD(Future, predecessor),
                typename pika::util::make_index_pack<sizeof...(Ts)>::type(),
                args_);
        }
    };

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    fused_bulk_sync_execute_helper<Result, std::decay_t<Executor>,
        std::decay_t<F>, Shape, std::decay_t<Args>>
    make_fused_bulk_sync_execute_helper(
        Executor&& exec, F&& f, Shape const& shape, Args&& args)
    {
        return fused_bulk_sync_execute_helper<Result, std::decay_t<Executor>,
            std::decay_t<F>, Shape, std::decay_t<Args>>{
            PIKA_FORWARD(Executor, exec), PIKA_FORWARD(F, f), shape,
            PIKA_FORWARD(Args, args)};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename Shape, typename Future,
        std::size_t... Is, typename... Ts>
    PIKA_FORCEINLINE auto fused_bulk_async_execute(Executor&& exec, F&& f,
        Shape const& shape, Future&& predecessor, pika::util::index_pack<Is...>,
        pika::tuple<Ts...> const& args)
        -> decltype(execution::bulk_async_execute(PIKA_FORWARD(Executor, exec),
            PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Future, predecessor),
            pika::get<Is>(args)...))
    {
        return execution::bulk_async_execute(PIKA_FORWARD(Executor, exec),
            PIKA_FORWARD(F, f), shape, PIKA_FORWARD(Future, predecessor),
            pika::get<Is>(args)...);
    }

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    struct fused_bulk_async_execute_helper;

    template <typename Result, typename Executor, typename F, typename Shape,
        typename... Ts>
    struct fused_bulk_async_execute_helper<Result, Executor, F, Shape,
        pika::tuple<Ts...>>
    {
        Executor exec_;
        F f_;
        Shape shape_;
        pika::tuple<Ts...> args_;

        template <typename Future>
        Result operator()(Future&& predecessor)
        {
            return fused_bulk_async_execute(exec_, f_, shape_,
                PIKA_FORWARD(Future, predecessor),
                typename pika::util::make_index_pack<sizeof...(Ts)>::type(),
                args_);
        }
    };

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    fused_bulk_async_execute_helper<Result, std::decay_t<Executor>,
        std::decay_t<F>, std::decay_t<Shape>, std::decay_t<Args>>
    make_fused_bulk_async_execute_helper(
        Executor&& exec, F&& f, Shape&& shape, Args&& args)
    {
        return fused_bulk_async_execute_helper<Result, std::decay_t<Executor>,
            std::decay_t<F>, std::decay_t<Shape>, std::decay_t<Args>>{
            PIKA_FORWARD(Executor, exec), PIKA_FORWARD(F, f),
            PIKA_FORWARD(Shape, shape), PIKA_FORWARD(Args, args)};
    }
}}}}    // namespace pika::parallel::execution::detail
