//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/allocator_support/traits/is_allocator.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/split.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>

#include <utility>

namespace pika::execution::experimental {
    inline constexpr struct ensure_started_t final
      : pika::functional::detail::tag_fallback<ensure_started_t>
    {
    private:
        template <typename Sender,
            PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender> &&
                !split_detail::is_split_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Sender&& sender)
        {
            return split_detail::split_sender<Sender,
                pika::util::internal_allocator<>,
                split_detail::submission_type::eager>{
                PIKA_FORWARD(Sender, sender), {}};
        }

        template <typename Sender, typename Allocator,
            PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender> &&
                !split_detail::is_split_sender_v<Sender> &&
                pika::traits::is_allocator_v<Allocator>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Sender&& sender, Allocator const& allocator)
        {
            return split_detail::split_sender<Sender, Allocator,
                split_detail::submission_type::eager>{
                PIKA_FORWARD(Sender, sender), allocator};
        }

        template <typename Sender, typename Allocator,
            PIKA_CONCEPT_REQUIRES_(split_detail::is_split_sender_v<Sender> &&
                (Sender::subm_type == split_detail::submission_type::eager) &&
                std::is_same_v<typename Sender::allocator_type,
                    std::decay_t<Allocator>>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Sender&& sender, Allocator const&)
        {
            return PIKA_FORWARD(Sender, sender);
        }

        template <typename Sender, typename Allocator,
            PIKA_CONCEPT_REQUIRES_(split_detail::is_split_sender_v<Sender> &&
                !((Sender::subm_type == split_detail::submission_type::eager) &&
                    std::is_same_v<typename Sender::allocator_type,
                        std::decay_t<Allocator>>) )>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Sender&& sender, Allocator const& allocator)
        {
            return split_detail::split_sender<Sender, Allocator,
                split_detail::submission_type::eager>{
                PIKA_FORWARD(Sender, sender), allocator};
        }

        template <typename Sender,
            PIKA_CONCEPT_REQUIRES_(split_detail::is_split_sender_v<Sender> &&
                (std::decay_t<Sender>::subm_type ==
                    split_detail::submission_type::eager))>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Sender&& sender)
        {
            return PIKA_FORWARD(Sender, sender);
        }

        template <typename Sender,
            PIKA_CONCEPT_REQUIRES_(split_detail::is_split_sender_v<Sender> &&
                (std::decay_t<Sender>::subm_type ==
                    split_detail::submission_type::lazy))>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Sender&& sender)
        {
            return split_detail::split_sender<Sender,
                pika::util::internal_allocator<>,
                split_detail::submission_type::eager>{
                PIKA_FORWARD(Sender, sender), {}};
        }

        template <typename Allocator = pika::util::internal_allocator<>,
            PIKA_CONCEPT_REQUIRES_(pika::traits::is_allocator_v<Allocator>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<ensure_started_t, Allocator>{
                allocator};
        }
    } ensure_started{};
}    // namespace pika::execution::experimental
