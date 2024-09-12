//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else
# include <pika/concepts/concepts.hpp>
# include <pika/execution/algorithms/detail/partial_algorithm.hpp>
# include <pika/execution/algorithms/schedule_from.hpp>
# include <pika/execution_base/completion_scheduler.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/functional/detail/tag_priority_invoke.hpp>

# include <utility>

namespace pika::execution::experimental {
    inline constexpr struct continues_on_t final
      : pika::functional::detail::tag_priority<continues_on_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename Scheduler,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                is_scheduler_v<Scheduler> &&
                pika::execution::experimental::detail::
                    is_completion_scheduler_tag_invocable_v<
                        pika::execution::experimental::set_value_t, Sender,
                        continues_on_t, Scheduler>)>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_override_invoke(continues_on_t, Sender&& sender, Scheduler&& scheduler)
        {
            auto completion_scheduler = pika::execution::experimental::get_completion_scheduler<
                pika::execution::experimental::set_value_t>(
                pika::execution::experimental::get_env(sender));
            return pika::functional::detail::tag_invoke(continues_on_t{},
                PIKA_MOVE(completion_scheduler), PIKA_FORWARD(Sender, sender),
                PIKA_FORWARD(Scheduler, scheduler));
        }

        // clang-format off
        template <typename Sender, typename Scheduler,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                is_scheduler_v<Scheduler>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(continues_on_t, Sender&& predecessor_sender, Scheduler&& scheduler)
        {
            return schedule_from(
                PIKA_FORWARD(Scheduler, scheduler), PIKA_FORWARD(Sender, predecessor_sender));
        }

        template <typename Scheduler>
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(continues_on_t, Scheduler&& scheduler)
        {
            return detail::partial_algorithm<continues_on_t, Scheduler>{
                PIKA_FORWARD(Scheduler, scheduler)};
        }
    } continues_on{};

    using transfer_t PIKA_DEPRECATED("transfer_t has been renamed continues_on_t") = continues_on_t;
    PIKA_DEPRECATED("transfer has been renamed continues_on")
    inline constexpr continues_on_t transfer{};
}    // namespace pika::execution::experimental
#endif
