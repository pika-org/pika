//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <pika/execution/algorithms/continues_on.hpp>
#include <pika/execution/algorithms/just.hpp>
#if !defined(PIKA_HAVE_STDEXEC)
# include <pika/functional/detail/tag_fallback_invoke.hpp>
#endif

#include <utility>

namespace pika::execution::experimental {
#if defined(PIKA_HAVE_STDEXEC)
    // pika uses its own transfer_just implementation because stdexec's transfer_just is deprecated.
    PIKA_DEPRECATED(
        "transfer_just will be removed in the future, use transfer and just separately instead")
    inline constexpr struct transfer_just_t
    {
        template <typename Scheduler, typename... Ts>
        constexpr PIKA_FORCEINLINE auto operator()(Scheduler&& scheduler, Ts&&... ts) const
        {
            return continues_on(just(std::forward<Ts>(ts)...), std::forward<Scheduler>(scheduler));
        }
    } transfer_just{};
#else
    PIKA_DEPRECATED(
        "transfer_just will be removed in the future, use transfer and just separately instead")
    inline constexpr struct transfer_just_t final
      : pika::functional::detail::tag_fallback<transfer_just_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(transfer_just_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return continues_on(just(std::forward<Ts>(ts)...), std::forward<Scheduler>(scheduler));
        }
    } transfer_just{};
#endif
}    // namespace pika::execution::experimental
