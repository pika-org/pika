//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/parallel/util/loop.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Sent, typename F>
    constexpr Iter sequential_generate_helper(Iter first, Sent last, F&& f)
    {
        return util::loop_ind(pika::execution::seq, first, last,
            [f = PIKA_FORWARD(F, f)](auto& v) mutable { v = f(); });
    }

    struct sequential_generate_t
      : pika::functional::detail::tag_fallback<sequential_generate_t>
    {
    private:
        template <typename ExPolicy, typename Iter, typename Sent, typename F>
        friend constexpr Iter tag_fallback_invoke(
            sequential_generate_t, ExPolicy&&, Iter first, Sent last, F&& f)
        {
            return sequential_generate_helper(first, last, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    inline constexpr sequential_generate_t sequential_generate =
        sequential_generate_t{};
#else
    template <typename ExPolicy, typename Iter, typename Sent, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE Iter sequential_generate(
        ExPolicy&& policy, Iter first, Sent last, F&& f)
    {
        return sequential_generate_t{}(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F>
    constexpr Iter sequential_generate_n_helper(
        Iter first, std::size_t count, F&& f)
    {
        return std::generate_n(first, count, f);
    }

    struct sequential_generate_n_t
      : pika::functional::detail::tag_fallback<sequential_generate_n_t>
    {
    private:
        template <typename ExPolicy, typename Iter, typename F>
        friend constexpr Iter tag_fallback_invoke(sequential_generate_n_t,
            ExPolicy&&, Iter first, std::size_t count, F&& f)
        {
            return sequential_generate_n_helper(
                first, count, PIKA_FORWARD(F, f));
        }
    };

#if !defined(PIKA_COMPUTE_DEVICE_CODE)
    inline constexpr sequential_generate_n_t sequential_generate_n =
        sequential_generate_n_t{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE Iter sequential_generate_n(
        ExPolicy&& policy, Iter first, std::size_t count, F&& f)
    {
        return sequential_generate_n_t{}(
            PIKA_FORWARD(ExPolicy, policy), first, count, PIKA_FORWARD(F, f));
    }
#endif

}}}}    // namespace pika::parallel::v1::detail
