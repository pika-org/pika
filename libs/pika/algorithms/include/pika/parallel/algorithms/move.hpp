//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/move.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    /// Moves the elements in the range [first, last), to another range
    /// beginning at \a dest. After this operation the elements in the
    /// moved-from range will still contain valid values of the appropriate
    /// type, but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first move assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the move assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a move algorithm returns a
    ///           \a  pika::future<FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a move algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           moved.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    move(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest);

    // clang-format off
}

#else // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/copy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/transfer.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/foreach_partitioner.hpp>
#include <pika/parallel/util/result_types.hpp>
#include <pika/parallel/util/transfer.hpp>
#include <pika/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // move
    namespace detail {
        /// \cond NOINTERNAL

        template <typename IterPair>
        struct move_pair
          : public detail::algorithm<detail::move_pair<IterPair>, IterPair>
        {
            move_pair()
              : move_pair::algorithm("move")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter>
            static constexpr std::enable_if_t<
                !pika::traits::is_random_access_iterator_v<InIter>,
                util::in_out_result<InIter, OutIter>>
            sequential(ExPolicy, InIter first, Sent last, OutIter dest)
            {
                return util::move(first, last, dest);
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter>
            static constexpr std::enable_if_t<
                pika::traits::is_random_access_iterator_v<InIter>,
                util::in_out_result<InIter, OutIter>>
            sequential(ExPolicy, InIter first, Sent last, OutIter dest)
            {
                return util::move_n(first, detail::distance(first, last), dest);
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(
                ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
            {
                typedef pika::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;

                return util::detail::get_in_out_result(
                    util::foreach_partitioner<ExPolicy>::call(
                        PIKA_FORWARD(ExPolicy, policy),
                        pika::util::make_zip_iterator(first, dest),
                        detail::distance(first, last),
                        [](zip_iterator part_begin, std::size_t part_size,
                            std::size_t) {
                            using pika::get;

                            auto iters = part_begin.get_iterator_tuple();
                            util::move_n(
                                get<0>(iters), part_size, get<1>(iters));
                        },
                        [](zip_iterator&& last) -> zip_iterator {
                            return PIKA_MOVE(last);
                        }));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter1, typename FwdIter2>
        struct move : public move_pair<util::in_out_result<FwdIter1, FwdIter2>>
        {
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator<FwdIter1>::value &&
            pika::traits::is_iterator<FwdIter2>::value)>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::move is deprecated, use pika::move instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        move(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
    {
        return detail::transfer<detail::move<FwdIter1, FwdIter2>>(
            PIKA_FORWARD(ExPolicy, policy), first, last, dest);
    }
}}}    // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::move
    inline constexpr struct move_t final
      : pika::detail::tag_parallel_algorithm<move_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter1>::value &&
                pika::traits::is_iterator<FwdIter2>::value)>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_invoke(move_t, ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest)
        {
            return pika::parallel::util::get_second_element(
                pika::parallel::v1::detail::transfer<
                    pika::parallel::v1::detail::move<FwdIter1, FwdIter2>>(
                    PIKA_FORWARD(ExPolicy, policy), first, last, dest));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter1>::value &&
                pika::traits::is_iterator<FwdIter2>::value)>
        // clang-format on
        friend FwdIter2 tag_fallback_invoke(
            move_t, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
        {
            return std::move(first, last, dest);
        }
    } move{};
}    // namespace pika

#endif    // DOXYGEN
