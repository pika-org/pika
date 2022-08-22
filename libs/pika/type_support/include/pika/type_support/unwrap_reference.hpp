//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <functional>

namespace pika::detail {
    template <typename T>
    struct unwrap_reference_impl
    {
        using type = T;
    };

    template <typename T>
    struct unwrap_reference_impl<std::reference_wrapper<T>>
    {
        using type = T;
    };

    template <typename T>
    struct unwrap_reference_impl<std::reference_wrapper<T> const>
    {
        using type = T;
    };

    template <typename T>
    PIKA_FORCEINLINE typename unwrap_reference_impl<T>::type& unwrap_reference(
        T& t)
    {
        return t;
    }
}    // namespace pika::detail
