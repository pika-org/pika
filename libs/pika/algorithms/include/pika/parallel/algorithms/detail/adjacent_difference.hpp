//  Copyright (c) 2021 Srinivas Yadav
//  Copyright (c) 2021 Karame M.shokooh
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {

    template <typename ExPolicy>
    struct sequential_adjacent_difference_t
      : pika::functional::detail::tag_fallback<
            sequential_adjacent_difference_t<ExPolicy>>
    {
    private:
        template <typename InIter, typename Sent, typename OutIter, typename Op>
        friend inline OutIter tag_fallback_invoke(
            sequential_adjacent_difference_t<ExPolicy>, InIter first, Sent last,
            OutIter dest, Op&& op)
        {
            if (first == last)
                return dest;

            using value_t = typename std::iterator_traits<InIter>::value_type;
            value_t acc = *first;
            *dest = acc;
            while (++first != last)
            {
                value_t val = *first;
                *++dest = op(val, PIKA_MOVE(acc));
                acc = PIKA_MOVE(val);
            }
            return ++dest;
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr sequential_adjacent_difference_t<ExPolicy>
        sequential_adjacent_difference =
            sequential_adjacent_difference_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename InIter, typename Sent,
        typename OutIter, typename Op>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE OutIter sequential_adjacent_difference(
        InIter first, Sent last, OutIter dest, Op&& op)
    {
        return sequential_adjacent_difference_t<ExPolicy>{}(
            first, last, dest, PIKA_FORWARD(Op, op));
    }
#endif

}}}}    // namespace pika::parallel::v1::detail
