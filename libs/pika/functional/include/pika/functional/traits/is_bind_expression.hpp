//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <functional>
#endif

namespace pika::detail {
    template <typename T>
    struct is_bind_expression : std::is_bind_expression<T>
    {
    };

    template <typename T>
    struct is_bind_expression<T const> : is_bind_expression<T>
    {
    };

    template <typename T>
    inline constexpr bool is_bind_expression_v = is_bind_expression<T>::value;
}    // namespace pika::detail
