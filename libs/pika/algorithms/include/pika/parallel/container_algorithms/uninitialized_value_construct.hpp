//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/uninitialized_value_construct.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika { namespace ranges {
    // clang-format off

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by value-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked without an execution policy object will execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a uninitialized_value_construct algorithm returns a
    ///           returns \a FwdIter.
    ///           The \a uninitialized_value_construct algorithm returns the
    ///           output iterator to the element in the range, one past
    ///           the last element constructed.
    ///
    template <typename FwdIter, typename Sent>
    FwdIter uninitialized_value_construct(
        FwdIter first, Sent last);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by value-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
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
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_value_construct algorithm returns a
    ///           \a pika::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_value_construct algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>::type
    uninitialized_value_construct(
        ExPolicy&& policy, FwdIter first, Sent last);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by value-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    ///
    /// \param rng          Refers to the range to which will be value
    ///                     constructed.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked without an execution policy object will execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a uninitialized_value_construct algorithm returns a
    ///           returns \a pika::traits::range_traits<Rng>
    ///           ::iterator_type.
    ///           The \a uninitialized_value_construct algorithm returns
    ///           the output iterator to the element in the range, one past
    ///           the last element constructed.
    ///
    template <typename Rng>
    typename pika::traits::range_traits<Rng>::iterator_type
    uninitialized_value_construct(Rng&& rng);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by value-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the range to which the value
    ///                     will be value consutrcted
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_value_construct algorithm returns a
    ///           \a pika::future<typename pika::traits::range_traits<Rng>
    ///           ::iterator_type>, if the
    ///           execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a typename
    ///           pika::traits::range_traits<Rng>::iterator_type otherwise.
    ///           The \a uninitialized_value_construct algorithm returns
    ///           the output iterator to the element in the range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename Rng>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        typename pika::traits::range_traits<Rng>::iterator_type>::type
    uninitialized_value_construct(ExPolicy&& policy, Rng&& rng);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// [first, first + count) by value-initialization. If an exception
    /// is thrown during the initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct_n
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a uninitialized_value_construct_n algorithm returns a
    ///           returns \a FwdIter.
    ///           The \a uninitialized_value_construct_n algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename FwdIter, typename Size>
    FwdIter uninitialized_value_construct_n(FwdIter first, Size count);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// [first, first + count) by value-initialization. If an exception
    /// is thrown during the initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_value_construct_n algorithm returns a
    ///           \a pika::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_value_construct_n algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size>
    typename typename parallel::util::detail::algorithm_result<ExPolicy,
        FwdIter>::type
    uninitialized_value_construct_n(
        ExPolicy&& policy, FwdIter first, Size count);

    // clang-format on
}}    // namespace pika::ranges
#else

#include <pika/local/config.hpp>
#include <pika/algorithms/traits/projected_range.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/algorithms/uninitialized_value_construct.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace ranges {
    inline constexpr struct uninitialized_value_construct_t final
      : pika::detail::tag_parallel_algorithm<uninitialized_value_construct_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_forward_iterator<FwdIter>::value &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            pika::ranges::uninitialized_value_construct_t, FwdIter first,
            Sent last)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::uninitialized_value_construct<
                FwdIter>()
                .call(pika::execution::seq, first, last);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_forward_iterator<FwdIter>::value &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(pika::ranges::uninitialized_value_construct_t,
            ExPolicy&& policy, FwdIter first, Sent last)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::uninitialized_value_construct<
                FwdIter>()
                .call(PIKA_FORWARD(ExPolicy, policy), first, last);
        }

        // clang-format off
        template <typename Rng,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename pika::traits::range_traits<Rng>::iterator_type
        tag_fallback_invoke(
            pika::ranges::uninitialized_value_construct_t, Rng&& rng)
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                pika::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::uninitialized_value_construct<
                iterator_type>()
                .call(pika::execution::seq, std::begin(rng), std::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename pika::traits::range_traits<Rng>::iterator_type>::type
        tag_fallback_invoke(pika::ranges::uninitialized_value_construct_t,
            ExPolicy&& policy, Rng&& rng)
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                pika::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::uninitialized_value_construct<
                iterator_type>()
                .call(PIKA_FORWARD(ExPolicy, policy), std::begin(rng),
                    std::end(rng));
        }
    } uninitialized_value_construct{};

    inline constexpr struct uninitialized_value_construct_n_t final
      : pika::detail::tag_parallel_algorithm<uninitialized_value_construct_n_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            pika::ranges::uninitialized_value_construct_n_t, FwdIter first,
            Size count)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::uninitialized_value_construct_n<
                FwdIter>()
                .call(pika::execution::seq, first, count);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(pika::ranges::uninitialized_value_construct_n_t,
            ExPolicy&& policy, FwdIter first, Size count)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::uninitialized_value_construct_n<
                FwdIter>()
                .call(PIKA_FORWARD(ExPolicy, policy), first, count);
        }
    } uninitialized_value_construct_n{};
}}    // namespace pika::ranges

#endif
