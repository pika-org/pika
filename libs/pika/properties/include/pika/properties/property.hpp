//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace experimental {

    inline constexpr struct prefer_t
      : pika::functional::detail::tag_fallback<prefer_t>
    {
        // clang-format off
        template <typename Tag, typename... Tn>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
                prefer_t, Tag const& tag, Tn&&... tn)
            noexcept(noexcept(tag(PIKA_FORWARD(Tn, tn)...)))
            -> decltype(tag(PIKA_FORWARD(Tn, tn)...))
        // clang-format on
        {
            return tag(PIKA_FORWARD(Tn, tn)...);
        }

        // clang-format off
        template <typename Tag, typename T0, typename... Tn>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
                prefer_t, Tag, T0&& t0, Tn&&...)
            noexcept(noexcept(PIKA_FORWARD(T0, t0)))
            -> std::enable_if_t<
                    !pika::functional::is_tag_invocable_v<
                        prefer_t, Tag, T0, Tn...> &&
                    !pika::is_invocable_v<Tag, T0, Tn...>,
                    decltype(PIKA_FORWARD(T0, t0))>
        // clang-format on
        {
            return PIKA_FORWARD(T0, t0);
        }
    } prefer{};
}}    // namespace pika::experimental
