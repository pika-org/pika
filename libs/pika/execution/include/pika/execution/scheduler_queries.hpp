//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/concepts/concepts.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/tag_invoke.hpp>

namespace pika::execution::experimental {
    enum class forward_progress_guarantee
    {
        concurrent,
        parallel,
        weakly_parallel
    };

    namespace scheduler_queries_detail {
        struct forwarding_scheduler_query_t
        {
            template <typename Query,
                PIKA_CONCEPT_REQUIRES_(
                    pika::functional::is_nothrow_tag_invocable_v<
                        forwarding_scheduler_query_t, Query const&>)>
            constexpr bool operator()(Query const& query) const noexcept
            {
                return pika::functional::tag_invoke(*this, query);
            }

            template <typename Query,
                PIKA_CONCEPT_REQUIRES_(
                    !pika::functional::is_nothrow_tag_invocable_v<
                        forwarding_scheduler_query_t, Query const&>)>
            constexpr bool operator()(Query const&) const noexcept
            {
                return false;
            }
        };

        struct get_forward_progress_guarantee_t
        {
            template <typename Scheduler,
                PIKA_CONCEPT_REQUIRES_(is_scheduler_v<Scheduler>&&
                        pika::functional::is_nothrow_tag_invocable_v<
                            get_forward_progress_guarantee_t,
                            Scheduler const&>)>
            constexpr forward_progress_guarantee operator()(
                Scheduler const& scheduler) const noexcept
            {
                return pika::functional::tag_invoke(*this, scheduler);
            }

            template <typename Scheduler,
                PIKA_CONCEPT_REQUIRES_(is_scheduler_v<Scheduler> &&
                    !pika::functional::is_nothrow_tag_invocable_v<
                        get_forward_progress_guarantee_t, Scheduler const&>)>
            constexpr forward_progress_guarantee operator()(
                Scheduler const&) const noexcept
            {
                return forward_progress_guarantee::weakly_parallel;
            }
        };
    }    // namespace scheduler_queries_detail

    using scheduler_queries_detail::forwarding_scheduler_query_t;
    using scheduler_queries_detail::get_forward_progress_guarantee_t;

    inline constexpr forwarding_scheduler_query_t forwarding_scheduler_query{};
    inline constexpr get_forward_progress_guarantee_t
        get_forward_progress_guarantee{};
}    // namespace pika::execution::experimental
