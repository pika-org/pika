//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/execution/algorithms/just.hpp>
#include <pika/execution/algorithms/transfer.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>

#include <utility>

namespace pika { namespace execution { namespace experimental {
    inline constexpr struct transfer_just_t final
      : pika::functional::detail::tag_fallback<transfer_just_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            transfer_just_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return transfer(just(PIKA_FORWARD(Ts, ts)...),
                PIKA_FORWARD(Scheduler, scheduler));
        }
    } transfer_just{};
}}}    // namespace pika::execution::experimental
