//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_base/apply.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution_base/traits/is_executor.hpp>
#include <pika/executors/parallel_executor.hpp>
#include <pika/functional/deferred_call.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // Define apply() overloads for plain local functions and function objects.
    // dispatching trait for pika::apply

    // launch a plain function/function object
    template <typename Func, typename Enable>
    struct apply_dispatch
    {
        template <typename F, typename... Ts>
        PIKA_FORCEINLINE static typename std::enable_if<
            traits::detail::is_deferred_invocable<F, Ts...>::value, bool>::type
        call(F&& f, Ts&&... ts)
        {
            execution::parallel_executor exec;
            exec.post(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            return false;
        }
    };

    // The overload for pika::apply taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel::execution::executor
    // threads::executor
    template <typename Executor>
    struct apply_dispatch<Executor,
        typename std::enable_if<traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value>::type>
    {
        template <typename Executor_, typename F, typename... Ts>
        PIKA_FORCEINLINE static decltype(auto) call(
            Executor_&& exec, F&& f, Ts&&... ts)
        {
            parallel::execution::post(PIKA_FORWARD(Executor_, exec),
                PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
            return false;
        }
    };
}}    // namespace pika::detail
