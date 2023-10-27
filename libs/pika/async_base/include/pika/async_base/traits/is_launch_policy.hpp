//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <type_traits>

namespace pika::detail {
    // forward declaration only
    struct policy_holder_base;

    template <typename Policy>
    struct is_launch_policy : std::is_base_of<policy_holder_base, std::decay_t<Policy>>
    {
    };

    template <typename Policy>
    inline constexpr bool is_launch_policy_v = is_launch_policy<Policy>::value;
}    // namespace pika::detail
