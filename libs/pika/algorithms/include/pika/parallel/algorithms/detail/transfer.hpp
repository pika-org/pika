//  Copyright (c) 2016 Minh-Khanh Do
//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/algorithms/traits/projected.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // transfer
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // parallel version
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename Sent1, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transfer_(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest)
        {
            return Algo().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, dest);
        }

        // Executes transfer algorithm on the elements in the range [first, last),
        // to another range beginning at \a dest.
        //
        // \note   Complexity: Performs exactly \a last - \a first transfer assignments.
        //
        //
        // \tparam Algo        The algorithm that is used to transfer the elements.
        //                     Should be pika::parallel::detail::copy or
        //                     pika::parallel::detail::move.
        // \tparam ExPolicy    The type of the execution policy to use (deduced).
        //                     It describes the manner in which the execution
        //                     of the algorithm may be parallelized and the manner
        //                     in which it executes the move assignments.
        // \tparam FwdIter1    The type of the source iterators used (deduced).
        //                     This iterator type must meet the requirements of an
        //                     forward iterator.
        // \tparam FwdIter2    The type of the iterator representing the
        //                     destination range (deduced).
        //                     This iterator type must meet the requirements of an
        //                     output iterator.
        //
        // \param policy       The execution policy to use for the scheduling of
        //                     the iterations.
        // \param first        Refers to the beginning of the sequence of elements
        //                     the algorithm will be applied to.
        // \param last         Refers to the end of the sequence of elements the
        //                     algorithm will be applied to.
        // \param dest         Refers to the beginning of the destination range.
        //
        // \returns  The \a transfer algorithm returns a \a pika::future<FwdIter2> if
        //           the execution policy is of type
        //           \a sequenced_task_policy or
        //           \a parallel_task_policy and
        //           returns \a FwdIter2 otherwise.
        //           The \a move algorithm returns the output iterator to the
        //           element in the destination range, one past the last element
        //           transferred.
        //
        // clang-format off
        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename Sent1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter1>::value &&
                pika::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                pika::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        transfer(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator<FwdIter2>::value ||
                    (pika::is_sequenced_execution_policy<ExPolicy>::value &&
                        pika::traits::is_output_iterator<FwdIter2>::value),
                "Requires at least forward iterator or sequential execution.");

            return transfer_<Algo>(
                PIKA_FORWARD(ExPolicy, policy), first, last, dest);
        }
    }    // namespace detail
}}}      // namespace pika::parallel::v1
