//  Copyright (c) 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace parallel { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    struct projection_identity
    {
        using is_transparent = std::true_type;

        template <typename T>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr T&& operator()(
            T&& val) const noexcept
        {
            return PIKA_FORWARD(T, val);
        }
    };
}}}    // namespace pika::parallel::util

namespace pika {

    // C++20 introduces std::identity
    using identity = pika::parallel::util::projection_identity;
}    // namespace pika
