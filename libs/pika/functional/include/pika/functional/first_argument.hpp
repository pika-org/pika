//  Copyright (c) 2017 Antoine Tran Tan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(PIKA_HAVE_MODULE)
#include <tuple>
#include <type_traits>
#endif

namespace pika::util::detail {
    template <typename Tuple>
    struct tuple_first_argument;

    template <>
    struct tuple_first_argument<std::tuple<>>
    {
        using type = std::false_type;
    };

    template <typename Arg0, typename... Args>
    struct tuple_first_argument<std::tuple<Arg0, Args...>>
    {
        using type = std::decay_t<Arg0>;
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F>
    struct function_first_argument;

    template <typename ReturnType>
    struct function_first_argument<ReturnType (*)()>
    {
        using type = std::false_type;
    };

    template <typename ReturnType, typename Arg0, typename... Args>
    struct function_first_argument<ReturnType (*)(Arg0, Args...)>
    {
        using type = std::decay_t<Arg0>;
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F>
    struct lambda_first_argument;

    template <typename ClassType, typename ReturnType>
    struct lambda_first_argument<ReturnType (ClassType::*)() const>
    {
        using type = std::false_type;
    };

    template <typename ClassType, typename ReturnType, typename Arg0, typename... Args>
    struct lambda_first_argument<ReturnType (ClassType::*)(Arg0, Args...) const>
    {
        using type = std::decay_t<Arg0>;
    };

    template <typename F, typename Enable = void>
    struct first_argument
    {
    };

    // Specialization for functions
    template <typename F>
    struct first_argument<F, std::enable_if_t<std::is_function_v<std::remove_pointer_t<F>>>>
      : detail::function_first_argument<F>
    {
    };

    // Specialization for lambdas
    template <typename F>
    struct first_argument<F, std::enable_if_t<!std::is_function_v<std::remove_pointer_t<F>>>>
      : detail::lambda_first_argument<decltype(&F::operator())>
    {
    };
}    // namespace pika::util::detail
