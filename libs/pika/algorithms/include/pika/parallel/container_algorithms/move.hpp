//  Copyright (c) 2017 Bruno Pitrus
//  Copyright (c) 2017-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/move.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    /// Moves the elements in the range \a rng to another range beginning
    /// at \a dest. After this operation the elements in the moved-from
    /// range will still contain valid values of the appropriate type,
    /// but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly
    ///         std::distance(begin(rng), end(rng)) assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the begin source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for FwdIter1.
    /// \tparam FwdIter     The type of the iterator representing the
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
    /// The assignments in the parallel \a copy algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a move algorithm returns a
    ///           \a pika::future<ranges::move_result<iterator_t<Rng>, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::move_result<iterator_t<Rng>, FwdIter2>
    ///           otherwise.
    ///           The \a move algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element moved.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter>
    typename util::detail::algorithm_result<
        ExPolicy, ranges::move_result<FwdIter1, FwdIter>>::type
    move(ExPolicy&& policy, FwdIter1 iter, Sent1 sent, FwdIter dest);

    /// Moves the elements in the range \a rng to another range beginning
    /// at \a dest. After this operation the elements in the moved-from
    /// range will still contain valid values of the appropriate type,
    /// but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly
    ///         std::distance(begin(rng), end(rng)) assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a move algorithm returns a
    ///           \a pika::future<ranges::move_result<iterator_t<Rng>, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::move_result<iterator_t<Rng>, FwdIter2>
    ///           otherwise.
    ///           The \a move algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element moved.
    ///
    template <typename ExPolicy, typename Rng, typename FwdIter>
    typename util::detail::algorithm_result<
        ExPolicy, ranges::move_result<
            typename pika::traits::range_traits<Rng>::iterator_type, FwdIter>
    >::type
    move(ExPolicy&& policy, Rng&& rng, FwdIter dest);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/traits/is_range.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <pika/parallel/algorithms/move.hpp>

#include <type_traits>
#include <utility>

namespace pika::ranges {
    template <typename I, typename O>
    using move_result = parallel::util::detail::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::ranges::move
    inline constexpr struct move_t final
      : pika::detail::tag_parallel_algorithm<move_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_sentinel_for<Sent1, Iter1>::value &&
                pika::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            move_result<Iter1, Iter2>>::type
        tag_fallback_invoke(
            move_t, ExPolicy&& policy, Iter1 first, Sent1 last, Iter2 dest)
        {
            return pika::parallel::detail::transfer<
                pika::parallel::detail::move<Iter1, Iter2>>(
                PIKA_FORWARD(ExPolicy, policy), first, last, dest);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Iter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value &&
                pika::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            move_result<typename pika::traits::range_iterator<Rng>::type,
                Iter2>>::type
        tag_fallback_invoke(move_t, ExPolicy&& policy, Rng&& rng, Iter2 dest)
        {
            using iterator_type =
                typename pika::traits::range_iterator<Rng>::type;

            return pika::parallel::detail::transfer<
                pika::parallel::detail::move<iterator_type, Iter2>>(
                PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
                pika::util::end(rng), dest);
        }

        // clang-format off
        template <typename Iter1, typename Sent1, typename Iter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_sentinel_for<Sent1, Iter1>::value &&
                pika::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend move_result<Iter1, Iter2> tag_fallback_invoke(
            move_t, Iter1 first, Sent1 last, Iter2 dest)
        {
            return pika::parallel::detail::transfer<
                pika::parallel::detail::move<Iter1, Iter2>>(
                pika::execution::seq, first, last, dest);
        }

        // clang-format off
        template <typename Rng, typename Iter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value &&
                pika::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend move_result<typename pika::traits::range_iterator<Rng>::type,
            Iter2>
        tag_fallback_invoke(move_t, Rng&& rng, Iter2 dest)
        {
            using iterator_type =
                typename pika::traits::range_iterator<Rng>::type;

            return pika::parallel::detail::transfer<
                pika::parallel::detail::move<iterator_type, Iter2>>(
                pika::execution::seq, pika::util::begin(rng),
                pika::util::end(rng), dest);
        }
    } move{};
}    // namespace pika::ranges
#endif    // DOXYGEN
