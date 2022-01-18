//  Copyright (c) 2015-2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/shift_left.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the beginning of the range. For every integer i in [0, last - first
    ///  - n), moves the element originally at position first + n + i to
    /// position first + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_left algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable.
    ///
    /// \returns  The \a shift_left algorithm returns \a FwdIter.
    ///           The \a shift_left algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename FwdIter, typename Sent, typename Size>
    FwdIter shift_left(FwdIter first, Sent last, Size n);

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the beginning of the range. For every integer i in [0, last - first
    ///  - n), moves the element originally at position first + n + i to
    /// position first + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_left algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignment operations in the parallel \a shift_left algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable.
    ///
    /// \returns  The \a shift_left algorithm returns a
    ///           \a pika::future<FwdIter> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a shift_left algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Size>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>
    shift_left(ExPolicy&& policy, FwdIter first, Sent last, Size n);

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the beginning of the range. For every integer i in [0, last - first
    ///  - n), moves the element originally at position first + n + i to
    /// position first + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam Rng         The type of the range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param rng          Refers to the range in which the elements
    ///                     will be shifted.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_left algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \note The type of dereferenced \a pika::traits::range_iterator_t<Rng>
    ///       must meet the requirements of \a MoveAssignable.
    ///
    /// \returns  The \a shift_left algorithm returns \a
    ///           pika::traits::range_iterator_t<Rng>.
    ///           The \a shift_left algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename Rng, typename Size>
    pika::traits::range_iterator_t<Rng> shift_left(Rng&& rng, Size n);

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the beginning of the range. For every integer i in [0, last - first
    ///  - n), moves the element originally at position first + n + i to
    /// position first + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the range in which the elements
    ///                     will be shifted.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_left algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignment operations in the parallel \a shift_left algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a pika::traits::range_iterator_t<Rng>
    ///       must meet the requirements of \a MoveAssignable.
    ///
    /// \returns  The \a shift_left algorithm returns a
    ///           \a pika::future<pika::traits::range_iterator_t<Rng>> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a pika::traits::range_iterator_t<Rng> otherwise.
    ///           The \a shift_left algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename ExPolicy, typename Rng, typename Size>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        pika::traits::range_iterator_t<Rng>>::type
    shift_left(ExPolicy&& policy, Rng&& rng, Size n);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_range.hpp>

#include <pika/algorithms/traits/projected_range.hpp>
#include <pika/parallel/algorithms/shift_left.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace ranges {
    inline constexpr struct shift_left_t final
      : pika::functional::detail::tag_fallback<shift_left_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter> &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            pika::ranges::shift_left_t, FwdIter first, Sent last, Size n)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_left<FwdIter>().call(
                pika::execution::seq, first, last, n);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator_v<FwdIter> &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(pika::ranges::shift_left_t, ExPolicy&& policy,
            FwdIter first, Sent last, Size n)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_left<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, n);
        }

        // clang-format off
        template <typename Rng, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend pika::traits::range_iterator_t<Rng> tag_fallback_invoke(
            pika::ranges::shift_left_t, Rng&& rng, Size n)
        {
            static_assert(pika::traits::is_forward_iterator_v<
                              pika::traits::range_iterator_t<Rng>>,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_left<
                pika::traits::range_iterator_t<Rng>>()
                .call(pika::execution::seq, std::begin(rng), std::end(rng), n);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            pika::traits::range_iterator_t<Rng>>::type
        tag_fallback_invoke(
            pika::ranges::shift_left_t, ExPolicy&& policy, Rng&& rng, Size n)
        {
            static_assert(pika::traits::is_forward_iterator_v<
                              pika::traits::range_iterator_t<Rng>>,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_left<
                pika::traits::range_iterator_t<Rng>>()
                .call(PIKA_FORWARD(ExPolicy, policy), std::begin(rng),
                    std::end(rng), n);
        }
    } shift_left{};
}}    // namespace pika::ranges

#endif    // DOXYGEN
