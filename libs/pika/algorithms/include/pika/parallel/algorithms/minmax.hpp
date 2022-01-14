//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: pikainspect:nominmax

/// \file parallel/algorithms/minmax.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {

    /////////////////////////////////////////////////////////////////////////////
    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    ///
    /// The comparisons in the parallel \a min_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a min_element algorithm returns \a FwdIter.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename FwdIter, typename F>
    FwdIter min_element(FwdIter first, FwdIter last, F&& f);

    /////////////////////////////////////////////////////////////////////////////
    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a min_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a min_element algorithm returns a \a pika::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    min_element(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f);

    /////////////////////////////////////////////////////////////////////////////
    /// Finds the largest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     This argument is optional and defaults to std::less.
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    ///
    /// The comparisons in the parallel \a min_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a max_element algorithm returns \a FwdIter.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename FwdIter, typename F>
    FwdIter max_element(FwdIter first, FwdIter last, F&& f);

    /////////////////////////////////////////////////////////////////////////////
    /// Removes all elements satisfying specific criteria from the range
    /// Finds the largest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a max_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     This argument is optional and defaults to std::less.
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a pika::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    max_element(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f);

    /////////////////////////////////////////////////////////////////////////////
    /// Finds the largest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     This argument is optional and defaults to std::less.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a
    ///           \a minmax_element_result<FwdIter>
    ///           The \a minmax_element algorithm returns a pair consisting of
    ///           an iterator to the smallest element as the min element and
    ///           an iterator to the largest element as the max element. Returns
    ///           minmax_element_result<FwdIter>{first, first} if the range is empty. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename FwdIter, typename F>
    minmax_element_result<FwdIter> minmax_element(
        FwdIter first, FwdIter last, F&& f);

    /////////////////////////////////////////////////////////////////////////////
    /// Finds the largest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a minmax_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     This argument is optional and defaults to std::less.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such that objects of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type1.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a
    /// \a minmax_element_result<FwdIter><FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a minmax_element_result<FwdIter>
    ///           otherwise.
    ///           The \a minmax_element algorithm returns a pair consisting of
    ///           an iterator to the smallest element as the min element and
    ///           an iterator to the largest element as the max element. Returns
    ///           std::make_pair(first, first) if the range is empty. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    typename util::detail::algorithm_result<ExPolicy,
        minmax_element_result<FwdIter>>::type
    minmax_element(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f);

}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <pika/algorithms/traits/is_value_proxy.hpp>
#include <pika/algorithms/traits/projected.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    template <typename T>
    using minmax_element_result = pika::parallel::util::min_max_result<T>;

    ///////////////////////////////////////////////////////////////////////////
    // min_element
    namespace detail {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj>
        constexpr FwdIter sequential_min_element(ExPolicy&&, FwdIter it,
            std::size_t count, F const& f, Proj const& proj)
        {
            if (count == 0 || count == 1)
                return it;

            using element_type = pika::traits::proxy_value_t<
                typename std::iterator_traits<FwdIter>::value_type>;

            auto smallest = it;

            element_type value = PIKA_INVOKE(proj, *smallest);
            util::loop_n<std::decay_t<ExPolicy>>(
                ++it, count - 1, [&](FwdIter const& curr) -> void {
                    element_type curr_value = PIKA_INVOKE(proj, *curr);
                    if (PIKA_INVOKE(f, curr_value, value))
                    {
                        smallest = curr;
                        value = PIKA_MOVE(curr_value);
                    }
                });

            return smallest;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        struct min_element : public detail::algorithm<min_element<Iter>, Iter>
        {
            // this has to be a member of the algorithm type as we access this
            // generically from the segmented algorithms
            template <typename ExPolicy, typename FwdIter, typename F,
                typename Proj>
            static constexpr pika::traits::proxy_value_t<
                typename std::iterator_traits<FwdIter>::value_type>
            sequential_minmax_element_ind(ExPolicy&&, FwdIter it,
                std::size_t count, F const& f, Proj const& proj)
            {
                PIKA_ASSERT(count != 0);

                if (count == 1)
                    return *it;

                auto smallest = *it;

                using element_type =
                    pika::traits::proxy_value_t<typename std::iterator_traits<
                        decltype(smallest)>::value_type>;

                element_type value = PIKA_INVOKE(proj, *smallest);
                util::loop_n<std::decay_t<ExPolicy>>(
                    ++it, count - 1, [&](FwdIter const& curr) -> void {
                        element_type curr_value = PIKA_INVOKE(proj, **curr);
                        if (PIKA_INVOKE(f, curr_value, value))
                        {
                            smallest = *curr;
                            value = PIKA_MOVE(curr_value);
                        }
                    });

                return smallest;
            }

            min_element()
              : min_element::algorithm("min_element")
            {
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static FwdIter sequential(
                ExPolicy&& policy, FwdIter first, Sent last, F&& f, Proj&& proj)
            {
                if (first == last)
                    return first;

                using element_type = pika::traits::proxy_value_t<
                    typename std::iterator_traits<FwdIter>::value_type>;

                auto smallest = first;

                element_type value = PIKA_INVOKE(proj, *smallest);
                util::loop(PIKA_FORWARD(ExPolicy, policy), ++first, last,
                    [&](FwdIter const& curr) -> void {
                        element_type curr_value = PIKA_INVOKE(proj, *curr);
                        if (PIKA_INVOKE(f, curr_value, value))
                        {
                            smallest = curr;
                            value = PIKA_MOVE(curr_value);
                        }
                    });

                return smallest;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& f,
                    Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        FwdIter>::get(PIKA_MOVE(first));
                }

                auto f1 = [f, proj, policy](
                              FwdIter it, std::size_t part_count) -> FwdIter {
                    return sequential_min_element(
                        policy, it, part_count, f, proj);
                };
                auto f2 = [policy, f = PIKA_FORWARD(F, f),
                              proj = PIKA_FORWARD(Proj, proj)](
                              std::vector<FwdIter>&& positions) -> FwdIter {
                    return min_element::sequential_minmax_element_ind(
                        policy, positions.begin(), positions.size(), f, proj);
                };

                return util::partitioner<ExPolicy, FwdIter, FwdIter>::call(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), PIKA_MOVE(f1),
                    pika::unwrapping(PIKA_MOVE(f2)));
            }
        };

        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F = detail::less,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy> &&
            pika::traits::is_iterator_v<FwdIter> &&
            traits::is_projected_v<Proj, FwdIter> &&
            traits::is_indirect_callable_v<
                ExPolicy, F,
                traits::projected<Proj, FwdIter>,
                traits::projected<Proj, FwdIter>
            >
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::min_element is deprecated, use pika::min_element "
        "instead") pika::parallel::util::detail::algorithm_result_t<ExPolicy,
        FwdIter> min_element(ExPolicy&& policy, FwdIter first, FwdIter last,
        F&& f = F(), Proj&& proj = Proj())
    {
        static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        return detail::min_element<FwdIter>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
            PIKA_FORWARD(Proj, proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // max_element
    namespace detail {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj>
        constexpr FwdIter sequential_max_element(ExPolicy&&, FwdIter it,
            std::size_t count, F const& f, Proj const& proj)
        {
            if (count == 0 || count == 1)
                return it;

            using element_type = pika::traits::proxy_value_t<
                typename std::iterator_traits<FwdIter>::value_type>;

            auto largest = it;

            element_type value = PIKA_INVOKE(proj, *largest);
            util::loop_n<std::decay_t<ExPolicy>>(
                ++it, count - 1, [&](FwdIter const& curr) -> void {
                    element_type curr_value = PIKA_INVOKE(proj, *curr);
                    if (!PIKA_INVOKE(f, curr_value, value))
                    {
                        largest = curr;
                        value = PIKA_MOVE(curr_value);
                    }
                });

            return largest;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        struct max_element : public detail::algorithm<max_element<Iter>, Iter>
        {
            // this has to be a member of the algorithm type as we access this
            // generically from the segmented algorithms
            template <typename ExPolicy, typename FwdIter, typename F,
                typename Proj>
            static constexpr typename std::iterator_traits<FwdIter>::value_type
            sequential_minmax_element_ind(ExPolicy&&, FwdIter it,
                std::size_t count, F const& f, Proj const& proj)
            {
                PIKA_ASSERT(count != 0);

                if (count == 1)
                    return *it;

                auto largest = *it;

                using element_type =
                    pika::traits::proxy_value_t<typename std::iterator_traits<
                        decltype(largest)>::value_type>;

                element_type value = PIKA_INVOKE(proj, *largest);
                util::loop_n<std::decay_t<ExPolicy>>(
                    ++it, count - 1, [&](FwdIter const& curr) -> void {
                        element_type curr_value = PIKA_INVOKE(proj, **curr);
                        if (!PIKA_INVOKE(f, curr_value, value))
                        {
                            largest = *curr;
                            value = PIKA_MOVE(curr_value);
                        }
                    });

                return largest;
            }

            max_element()
              : max_element::algorithm("max_element")
            {
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static FwdIter sequential(
                ExPolicy&& policy, FwdIter first, Sent last, F&& f, Proj&& proj)
            {
                if (first == last)
                    return first;

                using element_type = pika::traits::proxy_value_t<
                    typename std::iterator_traits<FwdIter>::value_type>;

                auto largest = first;

                element_type value = PIKA_INVOKE(proj, *largest);
                util::loop(PIKA_FORWARD(ExPolicy, policy), ++first, last,
                    [&](FwdIter const& curr) -> void {
                        element_type curr_value = PIKA_INVOKE(proj, *curr);
                        if (!PIKA_INVOKE(f, curr_value, value))
                        {
                            largest = curr;
                            value = PIKA_MOVE(curr_value);
                        }
                    });

                return largest;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& f,
                    Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        FwdIter>::get(PIKA_MOVE(first));
                }

                auto f1 = [f, proj, policy](
                              FwdIter it, std::size_t part_count) -> FwdIter {
                    return sequential_max_element(
                        policy, it, part_count, f, proj);
                };
                auto f2 = [policy, f = PIKA_FORWARD(F, f),
                              proj = PIKA_FORWARD(Proj, proj)](
                              std::vector<FwdIter>&& positions) -> FwdIter {
                    return max_element::sequential_minmax_element_ind(
                        policy, positions.begin(), positions.size(), f, proj);
                };

                return util::partitioner<ExPolicy, FwdIter, FwdIter>::call(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), PIKA_MOVE(f1),
                    pika::unwrapping(PIKA_MOVE(f2)));
            }
        };

        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F = detail::less,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy> &&
            pika::traits::is_iterator_v<FwdIter> &&
            traits::is_projected_v<Proj, FwdIter> &&
            traits::is_indirect_callable_v<
                ExPolicy, F,
                traits::projected<Proj, FwdIter>,
                traits::projected<Proj, FwdIter>
            >
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::max_element is deprecated, use pika::max_element "
        "instead") pika::parallel::util::detail::algorithm_result_t<ExPolicy,
        FwdIter> max_element(ExPolicy&& policy, FwdIter first, FwdIter last,
        F&& f = F(), Proj&& proj = Proj())
    {
        static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        return detail::max_element<FwdIter>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
            PIKA_FORWARD(Proj, proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // minmax_element
    namespace detail {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj>
        minmax_element_result<FwdIter> sequential_minmax_element(ExPolicy&&,
            FwdIter it, std::size_t count, F const& f, Proj const& proj)
        {
            minmax_element_result<FwdIter> result = {it, it};

            if (count == 0 || count == 1)
                return result;

            using element_type = pika::traits::proxy_value_t<
                typename std::iterator_traits<FwdIter>::value_type>;

            element_type min_value = PIKA_INVOKE(proj, *it);
            element_type max_value = min_value;
            util::loop_n<std::decay_t<ExPolicy>>(
                ++it, count - 1, [&](FwdIter const& curr) -> void {
                    element_type curr_value = PIKA_INVOKE(proj, *curr);
                    if (PIKA_INVOKE(f, curr_value, min_value))
                    {
                        result.min = curr;
                        min_value = curr_value;
                    }

                    if (!PIKA_INVOKE(f, curr_value, max_value))
                    {
                        result.max = curr;
                        max_value = PIKA_MOVE(curr_value);
                    }
                });

            return result;
        }

        template <typename Iter>
        struct minmax_element
          : public detail::algorithm<minmax_element<Iter>,
                minmax_element_result<Iter>>
        {
            // this has to be a member of the algorithm type as we access this
            // generically from the segmented algorithms
            template <typename ExPolicy, typename PairIter, typename F,
                typename Proj>
            static typename std::iterator_traits<PairIter>::value_type
            sequential_minmax_element_ind(ExPolicy&&, PairIter it,
                std::size_t count, F const& f, Proj const& proj)
            {
                PIKA_ASSERT(count != 0);

                if (count == 1)
                    return *it;

                using element_type = pika::traits::proxy_value_t<
                    typename std::iterator_traits<Iter>::value_type>;

                auto result = *it;

                element_type min_value = PIKA_INVOKE(proj, *result.min);
                element_type max_value = PIKA_INVOKE(proj, *result.max);
                util::loop_n<std::decay_t<ExPolicy>>(
                    ++it, count - 1, [&](PairIter const& curr) -> void {
                        element_type curr_min_value =
                            PIKA_INVOKE(proj, *curr->min);
                        if (PIKA_INVOKE(f, curr_min_value, min_value))
                        {
                            result.min = curr->min;
                            min_value = PIKA_MOVE(curr_min_value);
                        }

                        element_type curr_max_value =
                            PIKA_INVOKE(proj, *curr->max);
                        if (!PIKA_INVOKE(f, curr_max_value, max_value))
                        {
                            result.max = curr->max;
                            max_value = PIKA_MOVE(curr_max_value);
                        }
                    });

                return result;
            }

            minmax_element()
              : minmax_element::algorithm("minmax_element")
            {
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static minmax_element_result<FwdIter> sequential(
                ExPolicy&& policy, FwdIter first, Sent last, F&& f, Proj&& proj)
            {
                auto min = first, max = first;

                if (first == last || ++first == last)
                {
                    return minmax_element_result<FwdIter>{min, max};
                }

                using element_type = pika::traits::proxy_value_t<
                    typename std::iterator_traits<FwdIter>::value_type>;

                element_type min_value = PIKA_INVOKE(proj, *min);
                element_type max_value = PIKA_INVOKE(proj, *max);
                util::loop(PIKA_FORWARD(ExPolicy, policy), first, last,
                    [&](FwdIter const& curr) -> void {
                        element_type curr_value = PIKA_INVOKE(proj, *curr);
                        if (PIKA_INVOKE(f, curr_value, min_value))
                        {
                            min = curr;
                            min_value = curr_value;
                        }

                        if (!PIKA_INVOKE(f, curr_value, max_value))
                        {
                            max = curr;
                            max_value = PIKA_MOVE(curr_value);
                        }
                    });

                return minmax_element_result<FwdIter>{min, max};
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                minmax_element_result<FwdIter>>::type
            parallel(
                ExPolicy&& policy, FwdIter first, Sent last, F&& f, Proj&& proj)
            {
                typedef minmax_element_result<FwdIter> result_type;

                result_type result = {first, first};
                if (first == last || ++first == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        result_type>::get(PIKA_MOVE(result));
                }

                auto f1 = [f, proj, policy](FwdIter it, std::size_t part_count)
                    -> minmax_element_result<FwdIter> {
                    return sequential_minmax_element(
                        policy, it, part_count, f, proj);
                };
                auto f2 =
                    [policy, f = PIKA_FORWARD(F, f),
                        proj = PIKA_FORWARD(Proj, proj)](
                        std::vector<result_type>&& positions) -> result_type {
                    return minmax_element::sequential_minmax_element_ind(
                        policy, positions.begin(), positions.size(), f, proj);
                };

                return util::partitioner<ExPolicy, result_type,
                    result_type>::call(PIKA_FORWARD(ExPolicy, policy),
                    result.min, detail::distance(result.min, last),
                    PIKA_MOVE(f1), pika::unwrapping(PIKA_MOVE(f2)));
            }
        };

        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F = detail::less,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy> &&
            pika::traits::is_iterator_v<FwdIter> &&
            traits::is_projected_v<Proj, FwdIter> &&
            traits::is_indirect_callable_v<
                ExPolicy, F,
                traits::projected<Proj, FwdIter>,
                traits::projected<Proj, FwdIter>
            >
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::minmax_element is deprecated, use pika::minmax_element "
        "instead") pika::parallel::util::detail::algorithm_result_t<ExPolicy,
        minmax_element_result<FwdIter>> minmax_element(ExPolicy&& policy,
        FwdIter first, FwdIter last, F&& f = F(), Proj&& proj = Proj())
    {
        static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        return detail::minmax_element<FwdIter>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
            PIKA_FORWARD(Proj, proj));
    }
}}}    // namespace pika::parallel::v1

namespace pika {

    template <typename T>
    using minmax_element_result = pika::parallel::util::min_max_result<T>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::min_element
    inline constexpr struct min_element_t final
      : pika::detail::tag_parallel_algorithm<min_element_t>
    {
        // clang-format off
        template <typename FwdIter,
            typename F = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            pika::min_element_t, FwdIter first, FwdIter last, F&& f = F())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::min_element<FwdIter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename F = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter>
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter>
        tag_fallback_invoke(pika::min_element_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, F&& f = F())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::min_element<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity());
        }
    } min_element{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::max_element
    inline constexpr struct max_element_t final
      : pika::detail::tag_parallel_algorithm<max_element_t>
    {
        // clang-format off
        template <typename FwdIter,
            typename F = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            pika::max_element_t, FwdIter first, FwdIter last, F&& f = F())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::max_element<FwdIter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename F = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter>
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter>
        tag_fallback_invoke(pika::max_element_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, F&& f = F())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::max_element<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity());
        }
    } max_element{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::minmax_element
    inline constexpr struct minmax_element_t final
      : pika::detail::tag_parallel_algorithm<minmax_element_t>
    {
        // clang-format off
        template <typename FwdIter,
            typename F = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter>
            )>
        // clang-format on
        friend minmax_element_result<FwdIter> tag_fallback_invoke(
            pika::minmax_element_t, FwdIter first, FwdIter last, F&& f = F())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::minmax_element<FwdIter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename F = pika::parallel::v1::detail::less,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter>
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            minmax_element_result<FwdIter>>
        tag_fallback_invoke(pika::minmax_element_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, F&& f = F())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::minmax_element<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity());
        }
    } minmax_element{};
}    // namespace pika

#endif    // DOXYGEN
