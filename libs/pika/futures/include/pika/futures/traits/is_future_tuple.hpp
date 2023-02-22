//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/futures/traits/is_future.hpp>
#include <pika/type_support/pack.hpp>

#include <tuple>
#include <type_traits>

namespace pika::traits {
    template <typename Tuple, typename Enable = void>
    struct is_future_tuple : std::false_type
    {
    };

    template <typename... Ts>
    struct is_future_tuple<std::tuple<Ts...>> : util::detail::all_of<is_future<Ts>...>
    {
    };
}    // namespace pika::traits
