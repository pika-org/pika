//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace detail {
    // dispatch point used for sync implementations
    template <typename Func, typename Enable = void>
    struct sync_dispatch;
}}    // namespace pika::detail

namespace pika {
    template <typename F, typename... Ts>
    PIKA_FORCEINLINE auto sync(F&& f, Ts&&... ts)
        -> decltype(detail::sync_dispatch<std::decay_t<F>>::call(
            PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...))
    {
        return detail::sync_dispatch<std::decay_t<F>>::call(
            PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, ts)...);
    }
}    // namespace pika
