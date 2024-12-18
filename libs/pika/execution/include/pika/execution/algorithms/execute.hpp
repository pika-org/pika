//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else
# include <pika/execution/algorithms/start_detached.hpp>
# include <pika/execution/algorithms/then.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/functional/detail/tag_fallback_invoke.hpp>

# include <utility>

namespace pika::execution::experimental {
    inline constexpr struct execute_t final : pika::functional::detail::tag_fallback<execute_t>
    {
    private:
        template <typename Scheduler, typename F>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(execute_t, Scheduler&& scheduler, F&& f)
        {
            return start_detached(
                then(schedule(std::forward<Scheduler>(scheduler)), std::forward<F>(f)));
        }
    } execute{};
}    // namespace pika::execution::experimental
#endif
