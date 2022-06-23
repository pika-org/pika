//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/datastructures/tuple.hpp>

#include <type_traits>

namespace pika::traits::detail {
    template <typename T, typename Enable = void>
    struct is_tuple_like_impl : std::false_type
    {
    };

    template <typename T>
    struct is_tuple_like_impl<T,
        std::void_t<decltype(pika::tuple_size<T>::value)>> : std::true_type
    {
    };

    /// Deduces to a true type if the given parameter T
    /// has a specific tuple like size.
    template <typename T>
    struct is_tuple_like : is_tuple_like_impl<typename std::remove_cv<T>::type>
    {
    };

    template <typename T>
    inline constexpr bool is_tuple_like_v = is_tuple_like<T>::value;
}    // namespace pika::traits::detail
