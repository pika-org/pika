//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/string_util/bad_lexical_cast.hpp>

#include <fmt/format.h>

#include <string>
#include <type_traits>

namespace pika::detail {

    template <typename T, typename Enable = void>
    struct to_string_impl
    {
        static std::string call(T const& value)
        {
            return fmt::format("{}", value);
        }
    };

    template <typename T>
    struct to_string_impl<T, std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>>>
    {
        static std::string call(T const& value)
        {
            return std::to_string(value);
        }
    };

    template <typename T>
    std::string to_string(T const& v)
    {
        try
        {
            return to_string_impl<T>::call(v);
        }
        catch (...)
        {
            return throw_bad_lexical_cast<T, std::string>();
        }
    }

}    // namespace pika::detail
