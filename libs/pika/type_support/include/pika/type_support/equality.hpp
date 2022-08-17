//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2019 Austin McCartney
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <type_traits>
#include <utility>

namespace pika::traits::detail {

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename U, typename Enable = void>
    struct equality_result
    {
    };

    template <typename T, typename U>
    struct equality_result<T, U,
        std::void_t<decltype(std::declval<const T&>() ==
            std::declval<const U&>())>>
    {
        using type =
            decltype(std::declval<const T&>() == std::declval<const U&>());
    };

    template <typename T, typename U>
    using equality_result_t = typename equality_result<T, U>::type;

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename U, typename Enable = void>
    struct inequality_result
    {
    };

    template <typename T, typename U>
    struct inequality_result<T, U,
        std::void_t<decltype(std::declval<const T&>() !=
            std::declval<const U&>())>>
    {
        using type =
            decltype(std::declval<const T&>() != std::declval<const U&>());
    };

    template <typename T, typename U>
    using inequality_result_t = typename inequality_result<T, U>::type;

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename U, typename Enable = void>
    struct is_weakly_equality_comparable_with_impl : std::false_type
    {
    };

    template <typename T, typename U>
    struct is_weakly_equality_comparable_with_impl<T, U,
        std::void_t<equality_result_t<T, U>, equality_result_t<U, T>,
            inequality_result_t<T, U>, inequality_result_t<U, T>>>
      : std::true_type
    {
    };

    template <typename T, typename U>
    struct is_weakly_equality_comparable_with
      : is_weakly_equality_comparable_with_impl<std::decay_t<T>,
            std::decay_t<U>>
    {
    };

    template <typename T, typename U>
    inline constexpr bool is_weakly_equality_comparable_with_v =
        is_weakly_equality_comparable_with<T, U>::value;

    // for now is_equality_comparable is equivalent to its weak version
    template <typename T, typename U>
    struct is_equality_comparable_with
      : is_weakly_equality_comparable_with_impl<std::decay_t<T>,
            std::decay_t<U>>
    {
    };

    template <typename T, typename U>
    inline constexpr bool is_equality_comparable_with_v =
        is_equality_comparable_with<T, U>::value;

    template <typename T>
    struct is_equality_comparable
      : is_weakly_equality_comparable_with_impl<std::decay_t<T>,
            std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_equality_comparable_v =
        is_equality_comparable<T>::value;
}    // namespace pika::traits::detail
