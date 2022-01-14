//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/concepts/concepts.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/iterator_support/zip_iterator.hpp>
#include <pika/parallel/algorithms/detail/adjacent_difference.hpp>
#include <pika/parallel/datapar/iterator_helpers.hpp>
#include <pika/parallel/datapar/loop.hpp>
#include <pika/parallel/datapar/zip_iterator.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_adjacent_difference
    {
        template <typename InIter, typename OutIter, typename Op>
        static inline OutIter call(
            InIter first, InIter last, OutIter dest, Op&& op)
        {
            if (first == last)
                return dest;
            auto count = std::distance(first, last) - 1;

            InIter prev = first;
            *dest++ = *first++;

            if (count == 0)
            {
                return dest;
            }

            using pika::get;
            using pika::util::make_zip_iterator;
            util::loop_n<std::decay_t<ExPolicy>>(
                make_zip_iterator(first, prev, dest), count,
                [op](auto&& it) mutable {
                    get<2>(*it) = PIKA_INVOKE(op, get<0>(*it), get<1>(*it));
                });
            std::advance(dest, count);
            return dest;
        }
    };

    template <typename ExPolicy, typename InIter, typename OutIter, typename Op,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_vectorpack_execution_policy<ExPolicy>::value&&
                pika::parallel::util::detail::iterator_datapar_compatible<
                    InIter>::value)>
    inline OutIter tag_invoke(sequential_adjacent_difference_t<ExPolicy>,
        InIter first, InIter last, OutIter dest, Op&& op)
    {
        return datapar_adjacent_difference<ExPolicy>::call(
            first, last, dest, PIKA_FORWARD(Op, op));
    }
}}}}    // namespace pika::parallel::v1::detail
#endif
