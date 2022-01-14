//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#include <type_traits>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {
    // main template represents non-integral types (raises error)
    template <typename Size, typename Enable = void>
    struct is_negative_helper;

    // signed integral values may be negative
    template <typename T>
    struct is_negative_helper<T,
        typename std::enable_if<std::is_signed<T>::value>::type>
    {
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static bool call(T const& size)
        {
            return size < 0;
        }

        PIKA_HOST_DEVICE PIKA_FORCEINLINE static T abs(T const& val)
        {
            return val < 0 ? -val : val;
        }

        PIKA_HOST_DEVICE PIKA_FORCEINLINE static T negate(T const& val)
        {
            return -val;
        }
    };

    // unsigned integral values are never negative
    template <typename T>
    struct is_negative_helper<T,
        typename std::enable_if<std::is_unsigned<T>::value>::type>
    {
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static bool call(T const&)
        {
            return false;
        }

        PIKA_HOST_DEVICE PIKA_FORCEINLINE static T abs(T const& val)
        {
            return val;
        }

        PIKA_HOST_DEVICE PIKA_FORCEINLINE static T negate(T const& val)
        {
            return val;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE bool is_negative(T const& val)
    {
        return is_negative_helper<T>::call(val);
    }

    template <typename T>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE T abs(T const& val)
    {
        return is_negative_helper<T>::abs(val);
    }

    template <typename T>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE T negate(T const& val)
    {
        return is_negative_helper<T>::negate(val);
    }
}}}}    // namespace pika::parallel::v1::detail
