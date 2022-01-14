//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/nth_element.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika { namespace ranges {
    // clang-format off

    /// nth_element is a partial sorting algorithm that rearranges elements in
    /// [first, last) such that the element pointed at by nth is changed to
    /// whatever element would occur in that position if [first, last) were
    /// sorted and all of the elements before this new nth element are less
    /// than or equal to the elements after the new nth element.
    ///
    /// \note   Complexity: Linear in std::distance(first, last) on average.
    ///         O(N) applications of the predicate, and O(N log N) swaps,
    ///         where N = last - first.
    ///
    /// \tparam RandomIt    The type of the source begin, nth, and end
    ///                     iterators used (deduced). This iterator type must
    ///                     meet the requirements of a random access iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Pred        Comparison function object which returns true if
    ///                     the first argument is less than the second.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param nth          Refers to the iterator defining the sort partition
    ///                     point
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param pred         Specifies the comparison function object which
    ///                     returns true if the first argument is less than
    ///                     (i.e. is ordered before) the second.
    ///                     The signature of this
    ///                     comparison function should be equivalent to:
    ///                     \code
    ///                     bool cmp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type must be such that an object of
    ///                     type \a randomIt can be dereferenced and then
    ///                     implicitly converted to Type. This defaults
    ///                     to std::less<>.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///                     This defaults to projection_identity.
    ///
    /// The comparison operations in the parallel \a nth_element
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a nth_element algorithm returns returns \a
    ///           RandomIt.
    ///           The \a nth_element algorithm returns an iterator equal
    ///           to last.
    ///
    template <typename RandomIt, typename Sent, typename Pred, typename Proj>
    RandomIt nth_element(RandomIt first, RandomIt nth, Sent last, Pred&& pred,
        Proj&& proj);

    /// nth_element is a partial sorting algorithm that rearranges elements in
    /// [first, last) such that the element pointed at by nth is changed to
    /// whatever element would occur in that position if [first, last) were
    /// sorted and all of the elements before this new nth element are less
    /// than or equal to the elements after the new nth element.
    ///
    /// \note   Complexity: Linear in std::distance(first, last) on average.
    ///         O(N) applications of the predicate, and O(N log N) swaps,
    ///         where N = last - first.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandomIt    The type of the source begin, nth, and end
    ///                     iterators used (deduced). This iterator type must
    ///                     meet the requirements of a random access iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Pred        Comparison function object which returns true if
    ///                     the first argument is less than the second.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param nth          Refers to the iterator defining the sort partition
    ///                     point
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param pred         Specifies the comparison function object which
    ///                     returns true if the first argument is less than
    ///                     (i.e. is ordered before) the second.
    ///                     The signature of this
    ///                     comparison function should be equivalent to:
    ///                     \code
    ///                     bool cmp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type must be such that an object of
    ///                     type \a randomIt can be dereferenced and then
    ///                     implicitly converted to Type. This defaults
    ///                     to std::less<>.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///                     This defaults to projection_identity.
    ///
    /// The comparison operations in the parallel \a nth_element invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a nth_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition algorithm returns a \a
    ///           pika::future<RandomIt>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a RandomIt otherwise.
    ///           The \a nth_element algorithm returns an iterator equal
    ///           to last.
    ///
    template <typename ExPolicy, typename RandomIt, typename Sent,
        typename Pred, typename Proj>
    parallel::util::detail::algorithm_result_t<ExPolicy, RandomIt>
    nth_element(ExPolicy&& policy, RandomIt first, RandomIt nth,
        Sent last, Pred&& pred, Proj&& proj);

    /// nth_element is a partial sorting algorithm that rearranges elements in
    /// [first, last) such that the element pointed at by nth is changed to
    /// whatever element would occur in that position if [first, last) were
    /// sorted and all of the elements before this new nth element are less
    /// than or equal to the elements after the new nth element.
    ///
    /// \note   Complexity: Linear in std::distance(first, last) on average.
    ///         O(N) applications of the predicate, and O(N log N) swaps,
    ///         where N = last - first.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an random access iterator.
    /// \tparam Pred        Comparison function object which returns true if
    ///                     the first argument is less than the second.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param nth          Refers to the iterator defining the sort partition
    ///                     point
    /// \param pred         Specifies the comparison function object which
    ///                     returns true if the first argument is less than
    ///                     (i.e. is ordered before) the second.
    ///                     The signature of this
    ///                     comparison function should be equivalent to:
    ///                     \code
    ///                     bool cmp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type must be such that an object of
    ///                     type \a randomIt can be dereferenced and then
    ///                     implicitly converted to Type. This defaults
    ///                     to std::less<>.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///                     This defaults to projection_identity.
    ///
    /// The comparison operations in the parallel \a nth_element
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a nth_element algorithm returns returns \a
    ///           pika::traits::range_iterator_t<Rng>.
    ///           The \a nth_element algorithm returns an iterator equal
    ///           to last.
    ///
    template <typename Rng, typename Pred, typename Proj>
    pika::traits::range_iterator_t<Rng> nth_element(Rng&& rng,
        pika::traits::range_iterator_t<Rng> nth, Pred&& pred,
        Proj&& proj);

    /// nth_element is a partial sorting algorithm that rearranges elements in
    /// [first, last) such that the element pointed at by nth is changed to
    /// whatever element would occur in that position if [first, last) were
    /// sorted and all of the elements before this new nth element are less
    /// than or equal to the elements after the new nth element.
    ///
    /// \note   Complexity: Linear in std::distance(first, last) on average.
    ///         O(N) applications of the predicate, and O(N log N) swaps,
    ///         where N = last - first.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an random access iterator.
    /// \tparam Pred        Comparison function object which returns true if
    ///                     the first argument is less than the second.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param nth          Refers to the iterator defining the sort partition
    ///                     point
    /// \param pred         Specifies the comparison function object which
    ///                     returns true if the first argument is less than
    ///                     (i.e. is ordered before) the second.
    ///                     The signature of this
    ///                     comparison function should be equivalent to:
    ///                     \code
    ///                     bool cmp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type must be such that an object of
    ///                     type \a randomIt can be dereferenced and then
    ///                     implicitly converted to Type. This defaults
    ///                     to std::less<>.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///                     This defaults to projection_identity.
    ///
    /// The comparison operations in the parallel \a nth_element invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a nth_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition algorithm returns a \a
    ///           pika::future<pika::traits::range_iterator_t<Rng>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a pika::traits::range_iterator_t<Rng> otherwise.
    ///           The \a nth_element algorithm returns an iterator equal
    ///           to last.
    ///
    template <typename ExPolicy, typename Rng, typename Pred, typename Proj>
    parallel::util::detail::algorithm_result_t<ExPolicy,
        pika::traits::range_iterator_t<Rng>>
    nth_element(ExPolicy&& policy, Rng&& rng,
        pika::traits::range_iterator_t<Rng> nth,
        Pred&& pred, Proj&& proj);

    // clang-format on
}}       // namespace pika::ranges
#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/algorithms/traits/projected_range.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/algorithms/nth_element.hpp>
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
    inline constexpr struct nth_element_t final
      : pika::detail::tag_parallel_algorithm<nth_element_t>
    {
    private:
        // clang-format off
        template <typename RandomIt, typename Sent,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_random_access_iterator_v<RandomIt> &&
                pika::traits::is_sentinel_for_v<Sent, RandomIt> &&
                pika::parallel::traits::is_projected_v<Proj, RandomIt> &&
                pika::parallel::traits::is_indirect_callable_v<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected<Proj, RandomIt>,
                    pika::parallel::traits::projected<Proj, RandomIt>
                >
            )>
        // clang-format on
        friend RandomIt tag_fallback_invoke(pika::ranges::nth_element_t,
            RandomIt first, RandomIt nth, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_random_access_iterator_v<RandomIt>,
                "Requires at least random access iterator.");

            return pika::parallel::v1::detail::nth_element<RandomIt>().call(
                pika::execution::seq, first, nth, last, PIKA_FORWARD(Pred, pred),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename RandomIt, typename Sent,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_random_access_iterator_v<RandomIt> &&
                pika::traits::is_sentinel_for_v<Sent, RandomIt> &&
                pika::parallel::traits::is_projected_v<Proj, RandomIt> &&
                pika::parallel::traits::is_indirect_callable_v<
                    ExPolicy, Pred,
                    pika::parallel::traits::projected<Proj, RandomIt>,
                    pika::parallel::traits::projected<Proj, RandomIt>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy, RandomIt>
        tag_fallback_invoke(pika::ranges::nth_element_t, ExPolicy&& policy,
            RandomIt first, RandomIt nth, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_random_access_iterator_v<RandomIt>,
                "Requires at least random access iterator.");

            return pika::parallel::v1::detail::nth_element<RandomIt>().call(
                PIKA_FORWARD(ExPolicy, policy), first, nth, last,
                PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range_v<Rng> &&
                pika::parallel::traits::is_projected_range_v<Proj, Rng> &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend pika::traits::range_iterator_t<Rng> tag_fallback_invoke(
            pika::ranges::nth_element_t, Rng&& rng,
            pika::traits::range_iterator_t<Rng> nth, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            using iterator_type = pika::traits::range_iterator_t<Rng>;

            static_assert(
                pika::traits::is_random_access_iterator_v<iterator_type>,
                "Requires at least random access iterator.");

            return pika::parallel::v1::detail::nth_element<iterator_type>().call(
                pika::execution::seq, std::begin(rng), nth, std::end(rng),
                PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_range_v<Rng> &&
                pika::parallel::traits::is_projected_range_v<Proj, Rng> &&
                pika::parallel::traits::is_indirect_callable_v<
                    ExPolicy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            pika::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(pika::ranges::nth_element_t, ExPolicy&& policy,
            Rng&& rng, pika::traits::range_iterator_t<Rng> nth,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            using iterator_type = pika::traits::range_iterator_t<Rng>;

            static_assert(
                pika::traits::is_random_access_iterator_v<iterator_type>,
                "Requires at least random access iterator.");

            return pika::parallel::v1::detail::nth_element<iterator_type>().call(
                PIKA_FORWARD(ExPolicy, policy), std::begin(rng), nth,
                std::end(rng), PIKA_FORWARD(Pred, pred),
                PIKA_FORWARD(Proj, proj));
        }
    } nth_element{};
}}    // namespace pika::ranges

#endif
