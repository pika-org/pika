//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>

namespace pika { namespace traits {
    template <typename Action, typename Enable = void>
    struct is_action : std::false_type
    {
    };

    template <typename T>
    inline constexpr bool is_action_v = false;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct is_bound_action : std::false_type
    {
    };

    template <typename T>
    inline constexpr bool is_bound_action_v = false;
}}    // namespace pika::traits
