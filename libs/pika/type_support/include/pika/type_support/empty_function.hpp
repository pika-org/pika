//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace pika::detail {

    struct empty_function
    {
        template <typename... Ts>
        constexpr void PIKA_STATIC_CALL_OPERATOR(Ts&&...) noexcept
        {
        }
    };
}    // namespace pika::detail
