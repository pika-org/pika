//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/adjacent_find.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika { namespace ranges {
    /// Searches the range [first, last) for two consecutive identical elements.
    ///
    /// \note   Complexity: Exactly the smaller of (result - first) + 1 and
    ///                     (last - first) - 1 application of the predicate
    ///                     where \a result is the value returned
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Pred        The type of an optional function/function object to use.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param pred         The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The
    ///                     signature should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 must be such
    ///                     that objects of type \a FwdIter
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 .
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a adjacent_find algorithm returns an iterator to the
    ///           first of the identical elements. If no such elements are
    ///           found, \a last is returned.
    template <typename FwdIter, typename Sent,
        typename Proj = pika::parallel::util::projection_identity,
        typename Pred = detail::equal_to>
    FwdIter adjacent_find(
        FwdIter first, Sent last, Pred&& pred = Pred(), Proj&& proj = Proj());

    /// Searches the range [first, last) for two consecutive identical elements.
    /// This version uses the given binary predicate pred
    ///
    /// \note   Complexity: Exactly the smaller of (result - first) + 1 and
    ///                     (last - first) - 1 application of the predicate
    ///                     where \a result is the value returned
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param pred         The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The
    ///                     signature should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 must be such
    ///                     that objects of type \a FwdIter
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 .
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a adjacent_find invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a adjacent_find invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a adjacent_find algorithm returns a \a pika::future<InIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a InIter otherwise.
    ///           The \a adjacent_find algorithm returns an iterator to the
    ///           first of the identical elements. If no such elements are
    ///           found, \a last is returned.
    ///
    ///           This overload of \a adjacent_find is available if the user
    ///           decides to provide their algorithm their own binary
    ///           predicate \a pred.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Proj = pika::parallel::util::projection_identity,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    adjacent_find(ExPolicy&& policy, FwdIter first, Sent last,
        Pred&& pred = Pred(), Proj&& proj = Proj());

    /// Searches the range rng for two consecutive identical elements.
    ///
    /// \note   Complexity: Exactly the smaller of (result - std::begin(rng)) + 1
    ///                     and (std::begin(rng) - std::end(rng)) - 1 applications
    ///                     of the predicate where \a result is the value returned
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Pred        The type of an optional function/function object to use.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The
    ///                     signature should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 must be such
    ///                     that objects of type \a FwdIter
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 .
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a adjacent_find algorithm returns an iterator to the
    ///           first of the identical elements. If no such elements are
    ///           found, \a last is returned.
    template <typename Rng,
        typename Proj = pika::parallel::util::projection_identity,
        typename Pred = detail::equal_to>
    typename pika::traits::range_traits<Rng>::iterator_type adjacent_find(
        ExPolicy&& policy, Rng&& rng, Pred&& pred = Pred(),
        Proj&& proj = Proj());

    /// Searches the range rng for two consecutive identical elements.
    ///
    /// \note   Complexity: Exactly the smaller of (result - std::begin(rng)) + 1
    ///                     and (std::begin(rng) - std::end(rng)) - 1 applications
    ///                     of the predicate where \a result is the value returned
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The
    ///                     signature should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 must be such
    ///                     that objects of type \a FwdIter
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 .
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a adjacent_find invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a adjacent_find invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a adjacent_find algorithm returns a \a pika::future<InIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a InIter otherwise.
    ///           The \a adjacent_find algorithm returns an iterator to the
    ///           first of the identical elements. If no such elements are
    ///           found, \a last is returned.
    ///
    ///           This overload of \a adjacent_find is available if the user
    ///           decides to provide their algorithm their own binary
    ///           predicate \a pred.
    ///
    template <typename ExPolicy, typename Rng,
        typename Proj = pika::parallel::util::projection_identity,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy,
        typename pika::traits::range_traits<Rng>::iterator_type>::type
    adjacent_find(ExPolicy&& policy, Rng&& rng, Pred&& pred = Pred(),
        Proj&& proj = Proj());
}}    // namespace pika::ranges
#else

#include <pika/local/config.hpp>
#include <pika/algorithms/traits/projected_range.hpp>
#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/algorithms/adjacent_find.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace ranges {
    inline constexpr struct adjacent_find_t final
      : pika::detail::tag_parallel_algorithm<adjacent_find_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            typename Proj = pika::parallel::util::projection_identity,
            typename Pred = pika::parallel::v1::detail::equal_to,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_forward_iterator<FwdIter>::value &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value &&
                pika::parallel::traits::is_projected<Proj, FwdIter>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected<Proj, FwdIter>,
                    pika::parallel::traits::projected<Proj, FwdIter>
                >::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(pika::ranges::adjacent_find_t,
            FwdIter first, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            return pika::parallel::v1::detail::adjacent_find<FwdIter, FwdIter>()
                .call(pika::execution::seq, first, last, PIKA_FORWARD(Pred, pred),
                    PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Proj = pika::parallel::util::projection_identity,
            typename Pred = pika::parallel::v1::detail::equal_to,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_forward_iterator<FwdIter>::value &&
                pika::traits::is_sentinel_for<Sent, FwdIter>::value &&
                pika::parallel::traits::is_projected<Proj, FwdIter>::value &&
                pika::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    pika::parallel::traits::projected<Proj, FwdIter>,
                    pika::parallel::traits::projected<Proj, FwdIter>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(pika::ranges::adjacent_find_t, ExPolicy&& policy,
            FwdIter first, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            return pika::parallel::v1::detail::adjacent_find<FwdIter, FwdIter>()
                .call(PIKA_FORWARD(ExPolicy, policy), first, last,
                    PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename Proj = pika::parallel::util::projection_identity,
            typename Pred = pika::parallel::v1::detail::equal_to,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename pika::traits::range_traits<Rng>::iterator_type
        tag_fallback_invoke(pika::ranges::adjacent_find_t, Rng&& rng,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (pika::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::adjacent_find<iterator_type,
                iterator_type>()
                .call(pika::execution::seq, std::begin(rng), std::end(rng),
                    PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Proj = pika::parallel::util::projection_identity,
            typename Pred = pika::parallel::v1::detail::equal_to,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename pika::traits::range_traits<Rng>::iterator_type>::type
        tag_fallback_invoke(pika::ranges::adjacent_find_t, ExPolicy&& policy,
            Rng&& rng, Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (pika::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::adjacent_find<iterator_type,
                iterator_type>()
                .call(PIKA_FORWARD(ExPolicy, policy), std::begin(rng),
                    std::end(rng), PIKA_FORWARD(Pred, pred),
                    PIKA_FORWARD(Proj, proj));
        }
    } adjacent_find{};
}}    // namespace pika::ranges

#endif
