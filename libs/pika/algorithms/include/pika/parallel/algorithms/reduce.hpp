//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/reduce.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika {

    // clang-format off

    /// Returns GENERALIZED_SUM(f, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source begin and end iterators used
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 \a Ret must be
    ///                     such that an object of type \a FwdIter can be
    ///                     dereferenced and then implicitly converted to any
    ///                     of those types.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a pika::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum over the elements given by the input range
    ///           [first, last).
    ///
    /// \note   GENERALIZED_SUM(op, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(op, b1, ..., bK), GENERALIZED_SUM(op, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIter, typename T, typename F>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    reduce(ExPolicy&& policy, FwdIter first, FwdIter last, T init, F&& f);

    /// Returns GENERALIZED_SUM(+, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source begin and end iterators used
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a pika::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    reduce(ExPolicy&& policy, FwdIter first, FwdIter last, T init);

    /// Returns GENERALIZED_SUM(+, T(), *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source begin and end iterators used
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a pika::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns T otherwise (where T is the value_type of
    ///           \a FwdIter).
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   The type of the initial value (and the result type) \a T is
    ///         determined from the value_type of the used \a FwdIter.
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIter>
    typename util::detail::algorithm_result<ExPolicy,
        typename std::iterator_traits<FwdIter>::value_type
    >::type
    reduce(ExPolicy&& policy, FwdIter first, FwdIter last);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_sentinel_for.hpp>
#include <pika/pack_traversal/unwrap.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>

#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/accumulate.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // reduce
    namespace detail {
        /// \cond NOINTERNAL
        template <typename T>
        struct reduce : public detail::algorithm<reduce<T>, T>
        {
            reduce()
              : reduce::algorithm("reduce")
            {
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename T_, typename Reduce>
            static T sequential(
                ExPolicy, InIterB first, InIterE last, T_&& init, Reduce&& r)
            {
                return detail::accumulate(
                    first, last, PIKA_FORWARD(T_, init), PIKA_FORWARD(Reduce, r));
            }

            template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
                typename T_, typename Reduce>
            static typename util::detail::algorithm_result<ExPolicy, T>::type
            parallel(ExPolicy&& policy, FwdIterB first, FwdIterE last,
                T_&& init, Reduce&& r)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, T>::get(
                        PIKA_FORWARD(T_, init));
                }

                auto f1 = [r](FwdIterB part_begin, std::size_t part_size) -> T {
                    T val = *part_begin;
                    return util::accumulate_n(
                        ++part_begin, --part_size, PIKA_MOVE(val), r);
                };

                return util::partitioner<ExPolicy, T>::call(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), PIKA_MOVE(f1),
                    pika::unwrapping([init = PIKA_FORWARD(T_, init),
                                        r = PIKA_FORWARD(Reduce, r)](
                                        std::vector<T>&& results) -> T {
                        return util::accumulate_n(pika::util::begin(results),
                            pika::util::size(results), init, r);
                    }));
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename T, typename F,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_sentinel_for<FwdIterE, FwdIterB>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::reduce is deprecated, use pika::ranges::reduce "
        "instead") typename util::detail::algorithm_result<ExPolicy, T>::type
        reduce(ExPolicy&& policy, FwdIterB first, FwdIterE last, T init, F&& f)
    {
        static_assert(pika::traits::is_forward_iterator<FwdIterB>::value,
            "Requires at least forward iterator.");

        static_assert(pika::traits::is_forward_iterator<FwdIterE>::value,
            "Requires at least forward iterator.");
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::reduce<T>().call(PIKA_FORWARD(ExPolicy, policy), first,
            last, PIKA_MOVE(init), PIKA_FORWARD(F, f));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename T,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_sentinel_for<FwdIterE, FwdIterB>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::reduce is deprecated, use pika::ranges::reduce instead")
        typename util::detail::algorithm_result<ExPolicy, T>::type
        reduce(ExPolicy&& policy, FwdIterB first, FwdIterE last, T init)
    {
        static_assert(pika::traits::is_forward_iterator<FwdIterB>::value,
            "Requires at least forward iterator.");

        static_assert(pika::traits::is_forward_iterator<FwdIterE>::value,
            "Requires at least forward iterator.");
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::reduce<T>().call(PIKA_FORWARD(ExPolicy, policy), first,
            last, PIKA_MOVE(init), std::plus<T>());
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_sentinel_for<FwdIterE, FwdIterB>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::reduce is deprecated, use pika::ranges::reduce instead")
        typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIterB>::value_type>::type
        reduce(ExPolicy&& policy, FwdIterB first, FwdIterE last)
    {
        static_assert(pika::traits::is_forward_iterator<FwdIterB>::value,
            "Requires at least forward iterator.");

        static_assert(pika::traits::is_forward_iterator<FwdIterE>::value,
            "Requires at least forward iterator.");

        using value_type = typename std::iterator_traits<FwdIterB>::value_type;
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::reduce<value_type>().call(PIKA_FORWARD(ExPolicy, policy),
            first, last, value_type{}, std::plus<value_type>());
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::reduce
    inline constexpr struct reduce_t final
      : pika::detail::tag_parallel_algorithm<reduce_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            typename T = typename std::iterator_traits<FwdIter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_fallback_invoke(pika::reduce_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, T init, F&& f)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::reduce<T>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_MOVE(init),
                PIKA_FORWARD(F, f));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename T = typename std::iterator_traits<FwdIter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_fallback_invoke(pika::reduce_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, T init)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::reduce<T>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_MOVE(init),
                std::plus<T>{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIter>::value_type>::type
        tag_fallback_invoke(
            pika::reduce_t, ExPolicy&& policy, FwdIter first, FwdIter last)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            using value_type =
                typename std::iterator_traits<FwdIter>::value_type;

            return pika::parallel::v1::detail::reduce<value_type>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, value_type{},
                std::plus<value_type>{});
        }

        // clang-format off
        template <typename FwdIter, typename F,
            typename T = typename std::iterator_traits<FwdIter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend T tag_fallback_invoke(
            pika::reduce_t, FwdIter first, FwdIter last, T init, F&& f)
        {
            static_assert(pika::traits::is_input_iterator<FwdIter>::value,
                "Requires at least input iterator.");

            return pika::parallel::v1::detail::reduce<T>().call(
                pika::execution::seq, first, last, PIKA_MOVE(init),
                PIKA_FORWARD(F, f));
        }

        // clang-format off
        template <typename FwdIter,
            typename T = typename std::iterator_traits<FwdIter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend T tag_fallback_invoke(
            pika::reduce_t, FwdIter first, FwdIter last, T init)
        {
            static_assert(pika::traits::is_input_iterator<FwdIter>::value,
                "Requires at least input iterator.");

            return pika::parallel::v1::detail::reduce<T>().call(
                pika::execution::seq, first, last, PIKA_MOVE(init),
                std::plus<T>{});
        }

        // clang-format off
        template <typename FwdIter,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename std::iterator_traits<FwdIter>::value_type
        tag_fallback_invoke(pika::reduce_t, FwdIter first, FwdIter last)
        {
            static_assert(pika::traits::is_input_iterator<FwdIter>::value,
                "Requires at least input iterator.");

            using value_type =
                typename std::iterator_traits<FwdIter>::value_type;

            return pika::parallel::v1::detail::reduce<value_type>().call(
                pika::execution::seq, first, last, value_type{},
                std::plus<value_type>());
        }
    } reduce{};
}    // namespace pika

#endif    // DOXYGEN
