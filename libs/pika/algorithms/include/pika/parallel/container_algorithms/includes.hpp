//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/includes.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika { namespace ranges {
    // clang-format off

    /// Returns true if every element from the sorted range [first2, last2) is
    /// found within the sorted range [first1, last1). Also returns true if
    /// [first2, last2) is empty. The version expects both ranges to be sorted
    /// with the user supplied binary predicate \a f.
    ///
    /// \note   At most 2*(N1+N2-1) comparisons, where
    ///         N1 = std::distance(first1, last1) and
    ///         N2 = std::distance(first2, last2).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter1       The type of the source iterators used (deduced)
    ///                     representing the first sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the source iterators used (deduced)
    ///                     representing the second sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the end source iterators used (deduced)
    ///                     representing the second sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter2.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a includes requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as includes. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2 can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a includes algorithm returns a \a pika::future<bool> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a includes algorithm returns true every element from the
    ///           sorted range [first2, last2) is found within the sorted range
    ///           [first1, last1). Also returns true if [first2, last2) is empty.
    ///
    template <typename ExPolicy, typename Iter1, typename Sent1, typename Iter2,
        typename Sent2, typename Pred = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type>::type
    includes(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Returns true if every element from the sorted range [first2, last2) is
    /// found within the sorted range [first1, last1). Also returns true if
    /// [first2, last2) is empty. The version expects both ranges to be sorted
    /// with the user supplied binary predicate \a f.
    ///
    /// \note   At most 2*(N1+N2-1) comparisons, where
    ///         N1 = std::distance(first1, last1) and
    ///         N2 = std::distance(first2, last2).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a includes requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first sequence of elements the algorithm
    ///                     will be applied to.
    /// \param rng2         Refers to the second sequence of elements the algorithm
    ///                     will be applied to.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as includes. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2 can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a includes algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a includes algorithm returns a \a pika::future<bool> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a includes algorithm returns true every element from the
    ///           sorted range [first2, last2) is found within the sorted range
    ///           [first1, last1). Also returns true if [first2, last2) is empty.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type>::type
    includes(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    // clang-format on
}}    // namespace pika::ranges

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/traits/is_sentinel_for.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/algorithms/traits/projected_range.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/includes.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::ranges::includes
    inline constexpr struct includes_t final
      : pika::detail::tag_parallel_algorithm<includes_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj1 = pika::parallel::util::projection_identity,
            typename Proj2 = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_sentinel_for<Sent1, Iter1>::value &&
                pika::parallel::traits::is_projected<Proj1, Iter1>::value &&
                pika::traits::is_sentinel_for<Sent2, Iter2>::value &&
                pika::parallel::traits::is_projected<Proj2, Iter2>::value &&
                pika::parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    pika::parallel::traits::projected<Proj1, Iter1>,
                    pika::parallel::traits::projected<Proj2, Iter2>
                >::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(includes_t, ExPolicy&& policy, Iter1 first1,
            Sent1 last1, Iter2 first2, Sent2 last2, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            static_assert((pika::traits::is_forward_iterator<Iter1>::value),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator<Iter2>::value),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::includes().call(
                PIKA_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                PIKA_FORWARD(Pred, op), PIKA_FORWARD(Proj1, proj1),
                PIKA_FORWARD(Proj2, proj2));
        }

        // clang-format off
        template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj1 = pika::parallel::util::projection_identity,
            typename Proj2 = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_sentinel_for<Sent1, Iter1>::value &&
                pika::parallel::traits::is_projected<Proj1, Iter1>::value &&
                pika::traits::is_sentinel_for<Sent2, Iter2>::value &&
                pika::parallel::traits::is_projected<Proj2, Iter2>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected<Proj1, Iter1>,
                    pika::parallel::traits::projected<Proj2, Iter2>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(includes_t, Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            static_assert((pika::traits::is_forward_iterator<Iter1>::value),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator<Iter2>::value),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::includes().call(
                pika::execution::seq, first1, last1, first2, last2,
                PIKA_FORWARD(Pred, op), PIKA_FORWARD(Proj1, proj1),
                PIKA_FORWARD(Proj2, proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj1 = pika::parallel::util::projection_identity,
            typename Proj2 = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng1>::value &&
                pika::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                pika::traits::is_range<Rng2>::value &&
                pika::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                pika::parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    pika::parallel::traits::projected_range<Proj1, Rng1>,
                    pika::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(includes_t, ExPolicy&& policy, Rng1&& rng1,
            Rng2&& rng2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename pika::traits::range_iterator<Rng1>::type;
            using iterator_type2 =
                typename pika::traits::range_iterator<Rng2>::type;

            static_assert(
                (pika::traits::is_forward_iterator<iterator_type1>::value),
                "Requires at least forward iterator.");
            static_assert(
                (pika::traits::is_forward_iterator<iterator_type2>::value),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::includes().call(
                PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng1),
                pika::util::end(rng1), pika::util::begin(rng2),
                pika::util::end(rng2), PIKA_FORWARD(Pred, op),
                PIKA_FORWARD(Proj1, proj1), PIKA_FORWARD(Proj2, proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            typename Pred = pika::parallel::v1::detail::less,
            typename Proj1 = pika::parallel::util::projection_identity,
            typename Proj2 = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng1>::value &&
                pika::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                pika::traits::is_range<Rng2>::value &&
                pika::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected_range<Proj1, Rng1>,
                    pika::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(includes_t, Rng1&& rng1, Rng2&& rng2,
            Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename pika::traits::range_iterator<Rng1>::type;
            using iterator_type2 =
                typename pika::traits::range_iterator<Rng2>::type;

            static_assert(
                (pika::traits::is_forward_iterator<iterator_type1>::value),
                "Requires at least forward iterator.");
            static_assert(
                (pika::traits::is_forward_iterator<iterator_type2>::value),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::includes().call(
                pika::execution::seq, pika::util::begin(rng1),
                pika::util::end(rng1), pika::util::begin(rng2),
                pika::util::end(rng2), PIKA_FORWARD(Pred, op),
                PIKA_FORWARD(Proj1, proj1), PIKA_FORWARD(Proj2, proj2));
        }
    } includes{};
}}    // namespace pika::ranges

#endif    // DOXYGEN
