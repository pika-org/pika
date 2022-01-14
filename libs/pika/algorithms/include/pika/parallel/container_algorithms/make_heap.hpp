//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/make_heap.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika { namespace ranges {
    // clang-format off

    /// Constructs a \a max \a heap in the range [first, last).
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param comp         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second. The signature of the function should be
    ///                     equivalent to
    ///                     \code
    ///                     bool comp(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a RndIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a
    ///           \a pika::future<Iter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a Iter
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename Comp,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        typename pika::traits::range_iterator<Rng>::type>::type
    make_heap(ExPolicy&& policy, Rng&& rng, Comp&& comp, Proj&& proj = Proj{});

    /// Constructs a \a max \a heap in the range [first, last). Uses the
    /// operator \a < for comparisons.
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a \a pika::future<void>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename Rng,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        typename pika::traits::range_iterator<Rng>::type>::type
    make_heap(ExPolicy&& policy, Rng&& rng, Proj&& proj = Proj{});

    // clang-format on
}}    // namespace pika::ranges

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/traits/is_range.hpp>
#include <pika/iterator_support/traits/is_sentinel_for.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/algorithms/traits/projected_range.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/make_heap.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::ranges::make_heap
    inline constexpr struct make_heap_t final
      : pika::detail::tag_parallel_algorithm<make_heap_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_sentinel_for<Sent, Iter>::value &&
                pika::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    pika::parallel::traits::projected<Proj, Iter>,
                    pika::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_invoke(make_heap_t, ExPolicy&& policy, Iter first,
            Sent last, Comp&& comp, Proj&& proj = Proj{})
        {
            static_assert(pika::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            return pika::parallel::v1::detail::make_heap<Iter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last,
                PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Comp,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            typename pika::traits::range_iterator<Rng>::type>::type
        tag_fallback_invoke(make_heap_t, ExPolicy&& policy, Rng& rng,
            Comp&& comp, Proj&& proj = Proj{})
        {
            using iterator_type =
                typename pika::traits::range_iterator<Rng>::type;

            static_assert(
                pika::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            return pika::parallel::v1::detail::make_heap<iterator_type>().call(
                PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
                pika::util::end(rng), PIKA_FORWARD(Comp, comp),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_sentinel_for<Sent, Iter>::value &&
                pika::parallel::traits::is_indirect_callable<ExPolicy,
                    std::less<typename std::iterator_traits<Iter>::value_type>,
                    pika::parallel::traits::projected<Proj, Iter>,
                    pika::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_invoke(make_heap_t, ExPolicy&& policy, Iter first,
            Sent last, Proj&& proj = Proj{})
        {
            static_assert(pika::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            using value_type = typename std::iterator_traits<Iter>::value_type;

            return pika::parallel::v1::detail::make_heap<Iter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last,
                std::less<value_type>(), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<ExPolicy,
                    std::less<typename std::iterator_traits<
                        typename pika::traits::range_iterator<Rng>::type
                    >::value_type>,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            typename pika::traits::range_iterator<Rng>::type>::type
        tag_fallback_invoke(
            make_heap_t, ExPolicy&& policy, Rng&& rng, Proj&& proj = Proj{})
        {
            using iterator_type =
                typename pika::traits::range_iterator<Rng>::type;

            static_assert(
                pika::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            return pika::parallel::v1::detail::make_heap<iterator_type>().call(
                PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
                pika::util::end(rng), std::less<value_type>(),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Iter, typename Sent, typename Comp,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_sentinel_for<Sent, Iter>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Comp,
                    pika::parallel::traits::projected<Proj, Iter>,
                    pika::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend Iter tag_fallback_invoke(make_heap_t, Iter first, Sent last,
            Comp&& comp, Proj&& proj = Proj{})
        {
            static_assert(pika::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            return pika::parallel::v1::detail::make_heap<Iter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(Comp, comp),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng, typename Comp,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy, Comp,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename pika::traits::range_iterator<Rng>::type
        tag_fallback_invoke(
            make_heap_t, Rng& rng, Comp&& comp, Proj&& proj = Proj{})
        {
            using iterator_type =
                typename pika::traits::range_iterator<Rng>::type;

            static_assert(
                pika::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            return pika::parallel::v1::detail::make_heap<iterator_type>().call(
                pika::execution::seq, pika::util::begin(rng), pika::util::end(rng),
                PIKA_FORWARD(Comp, comp), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Iter, typename Sent,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_sentinel_for<Sent, Iter>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy,
                    std::less<typename std::iterator_traits<Iter>::value_type>,
                    pika::parallel::traits::projected<Proj, Iter>,
                    pika::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend Iter tag_fallback_invoke(
            make_heap_t, Iter first, Sent last, Proj&& proj = Proj{})
        {
            static_assert(pika::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            using value_type = typename std::iterator_traits<Iter>::value_type;

            return pika::parallel::v1::detail::make_heap<Iter>().call(
                pika::execution::seq, first, last, std::less<value_type>(),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename Proj = pika::parallel::util::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::parallel::traits::is_projected_range<Proj, Rng>::value &&
                pika::parallel::traits::is_indirect_callable<
                    pika::execution::sequenced_policy,
                    std::less<typename std::iterator_traits<
                        typename pika::traits::range_iterator<Rng>::type
                    >::value_type>,
                    pika::parallel::traits::projected_range<Proj, Rng>,
                    pika::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename pika::traits::range_iterator<Rng>::type
        tag_fallback_invoke(make_heap_t, Rng&& rng, Proj&& proj = Proj{})
        {
            using iterator_type =
                typename pika::traits::range_iterator<Rng>::type;

            static_assert(
                pika::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            return pika::parallel::v1::detail::make_heap<iterator_type>().call(
                pika::execution::seq, pika::util::begin(rng), pika::util::end(rng),
                std::less<value_type>(), PIKA_FORWARD(Proj, proj));
        }
    } make_heap{};
}}    // namespace pika::ranges

#endif    // DOXYGEN
