//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>

#include <type_traits>
#include <utility>

namespace pika::experimental {

    inline constexpr struct prefer_t : pika::functional::detail::tag_fallback<prefer_t>
    {
        // clang-format off
        template <typename Tag, typename... Tn>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
                prefer_t, Tag const& tag, Tn&&... tn)
            noexcept(noexcept(tag(std::forward<Tn>(tn)...)))
            -> decltype(tag(std::forward<Tn>(tn)...))
        // clang-format on
        {
            return tag(std::forward<Tn>(tn)...);
        }

        // clang-format off
        template <typename Tag, typename T0, typename... Tn>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
                prefer_t, Tag, T0&& t0, Tn&&...)
            noexcept(noexcept(std::forward<T0>(t0)))
            -> std::enable_if_t<
                    !pika::functional::detail::is_tag_invocable_v<
                        prefer_t, Tag, T0, Tn...> &&
                    !std::is_invocable_v<Tag, T0, Tn...>,
                    decltype(std::forward<T0>(t0))>
        // clang-format on
        {
            return std::forward<T0>(t0);
        }
    } prefer{};
}    // namespace pika::experimental
