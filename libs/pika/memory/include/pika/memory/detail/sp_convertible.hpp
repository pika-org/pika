//  detail/sp_convertible.hpp
//
//  Copyright 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <cstddef>
#endif

namespace pika::memory::detail {

    template <typename Y, typename T>
    struct sp_convertible
    {
        using yes = char (&)[1];
        using no = char (&)[2];

        static yes f(T*);
        static no f(...);

        static constexpr bool value = sizeof((f) (static_cast<Y*>(nullptr))) == sizeof(yes);
    };

    template <typename Y, typename T>
    struct sp_convertible<Y, T[]>
    {
        static constexpr bool value = false;
    };

    template <typename Y, typename T>
    struct sp_convertible<Y[], T[]>
    {
        static constexpr bool value = sp_convertible<Y[1], T[1]>::value;
    };

    template <typename Y, std::size_t N, typename T>
    struct sp_convertible<Y[N], T[]>
    {
        static constexpr bool value = sp_convertible<Y[1], T[1]>::value;
    };

    template <typename Y, typename T>
    inline constexpr bool sp_convertible_v = sp_convertible<Y, T>::value;

}    // namespace pika::memory::detail
