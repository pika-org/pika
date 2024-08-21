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
# include <pika/execution/algorithms/transfer.hpp>
# include <pika/execution/algorithms/when_all.hpp>

#if !defined(PIKA_HAVE_MODULE)
# include <pika/functional/detail/tag_fallback_invoke.hpp>

# include <utility>
#endif

namespace pika::execution::experimental {
    inline constexpr struct transfer_when_all_t final
      : pika::functional::detail::tag_fallback<transfer_when_all_t>
    {
    private:
        template <typename Scheduler, typename... Ts>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(transfer_when_all_t, Scheduler&& scheduler, Ts&&... ts)
        {
            return transfer(when_all(PIKA_FORWARD(Ts, ts)...), PIKA_FORWARD(Scheduler, scheduler));
        }
    } transfer_when_all{};
}    // namespace pika::execution::experimental
#endif
