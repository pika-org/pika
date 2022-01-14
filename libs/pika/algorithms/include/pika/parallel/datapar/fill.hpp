//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/parallel/algorithms/detail/fill.hpp>
#include <pika/parallel/datapar/loop.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_fill
    {
        template <typename ExPolicy, typename Iter, typename Sent, typename T>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
            util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&& policy, Iter first, Sent last, T const& val)
        {
            pika::parallel::util::loop_ind(PIKA_FORWARD(ExPolicy, policy), first,
                last, [&val](auto& v) { v = val; });
            return first;
        }
    };

    template <typename ExPolicy, typename Iter, typename Sent, typename T,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_vectorpack_execution_policy<ExPolicy>::value&&
                pika::parallel::util::detail::iterator_datapar_compatible<
                    Iter>::value)>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE Iter tag_invoke(sequential_fill_t,
        ExPolicy&& policy, Iter first, Sent last, T const& value)
    {
        return datapar_fill::call(
            PIKA_FORWARD(ExPolicy, policy), first, last, value);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_fill_n
    {
        template <typename ExPolicy, typename Iter, typename T>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
            util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&&, Iter first, std::size_t count, T const& val)
        {
            pika::parallel::util::loop_n_ind<std::decay_t<ExPolicy>>(
                first, count, [&val](auto& v) { v = val; });
            return first;
        }
    };

    template <typename ExPolicy, typename Iter, typename T,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_vectorpack_execution_policy<ExPolicy>::value&&
                pika::parallel::util::detail::iterator_datapar_compatible<
                    Iter>::value)>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE Iter tag_invoke(sequential_fill_n_t,
        ExPolicy&& policy, Iter first, std::size_t count, T const& value)
    {
        return datapar_fill_n::call(
            PIKA_FORWARD(ExPolicy, policy), first, count, value);
    }
}}}}    // namespace pika::parallel::v1::detail
#endif
