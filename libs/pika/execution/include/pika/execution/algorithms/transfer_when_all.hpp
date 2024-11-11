//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else
# include <pika/execution/algorithms/continues_on.hpp>
# include <pika/execution/algorithms/when_all.hpp>
# include <pika/functional/detail/tag_fallback_invoke.hpp>

# include <utility>

namespace pika::execution::experimental {
    PIKA_DEPRECATED("transfer_when_all will be removed in the future, use transfer and when_all "
                    "separately instead")
    inline constexpr struct transfer_when_all_t final
      : pika::functional::detail::tag_fallback<transfer_when_all_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(transfer_when_all_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return continues_on(
                when_all(std::forward<Ts>(ts)...), std::forward<Scheduler>(scheduler));
        }
    } transfer_when_all{};
}    // namespace pika::execution::experimental
#endif
