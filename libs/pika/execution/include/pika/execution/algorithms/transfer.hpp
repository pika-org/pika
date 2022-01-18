//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/completion_scheduler.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_priority_invoke.hpp>

#include <utility>

namespace pika { namespace execution { namespace experimental {
    inline constexpr struct transfer_t final
      : pika::functional::detail::tag_priority<transfer_t>
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
                        transfer_t, Scheduler>)>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_override_invoke(
            transfer_t, Sender&& sender, Scheduler&& scheduler)
        {
            auto completion_scheduler =
                pika::execution::experimental::get_completion_scheduler<
                    pika::execution::experimental::set_value_t>(sender);
            return pika::functional::tag_invoke(transfer_t{},
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
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            transfer_t, Sender&& predecessor_sender, Scheduler&& scheduler)
        {
            return schedule_from(PIKA_FORWARD(Scheduler, scheduler),
                PIKA_FORWARD(Sender, predecessor_sender));
        }

        template <typename Scheduler>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            transfer_t, Scheduler&& scheduler)
        {
            return detail::partial_algorithm<transfer_t, Scheduler>{
                PIKA_FORWARD(Scheduler, scheduler)};
        }
    } transfer{};
}}}    // namespace pika::execution::experimental
