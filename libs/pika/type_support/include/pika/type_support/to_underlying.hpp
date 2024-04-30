//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>

namespace pika::detail {
    template <typename E>
    constexpr std::underlying_type_t<E> to_underlying(E e) noexcept
    {
        return static_cast<std::underlying_type_t<E>>(e);
    }
}    // namespace pika::detail
