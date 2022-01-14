//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/all_any_none.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    ///  Checks if unary predicate \a f returns true for no elements in the
    ///  range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a none_of requires \a F to meet the
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
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a none_of algorithm returns a \a pika::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a bool
    ///           otherwise.
    ///           The \a none_of algorithm returns true if the unary predicate
    ///           \a f returns true for no elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    none_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
        Proj&& proj = Proj());

    ///  Checks if unary predicate \a f returns true for at least one element
    ///  in the range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a any_of requires \a F to meet the
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
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a any_of algorithm returns a \a pika::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a any_of algorithm returns true if the unary predicate
    ///           \a f returns true for at least one element in the range,
    ///           false otherwise. It returns false if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    any_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
        Proj&& proj = Proj());

    /// Checks if unary predicate \a f returns true for all elements in the
    /// range [first, last).
    ///
    /// \note   Complexity: At most \a last - \a first applications of the
    ///         predicate \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a all_of requires \a F to meet the
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
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed
    ///                     to it. The type \a Type must be such that an object
    ///                     of type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a all_of algorithm returns a \a pika::future<bool> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a all_of algorithm returns true if the unary predicate
    ///           \a f returns true for all elements in the range, false
    ///           otherwise. It returns true if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, bool>::type
    all_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
        Proj&& proj = Proj());

    // clang-format on
}    // namespace pika
#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/type_support/void_guard.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/algorithms/detail/find.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/invoke_projected.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>
#include <pika/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {

    ///////////////////////////////////////////////////////////////////////////
    // none_of
    namespace detail {
        /// \cond NOINTERNAL
        struct none_of : public detail::algorithm<none_of, bool>
        {
            none_of()
              : none_of::algorithm("none_of")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj>
            static bool sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj)
            {
                return detail::sequential_find_if<ExPolicy>(first, last,
                           util::invoke_projected<F, Proj>(PIKA_FORWARD(F, f),
                               PIKA_FORWARD(Proj, proj))) == last;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& op,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                util::cancellation_token<> tok;
                auto f1 = [op = PIKA_FORWARD(F, op), tok,
                              proj = PIKA_FORWARD(Proj, proj)](
                              FwdIter part_begin,
                              std::size_t part_count) mutable -> bool {
                    detail::sequential_find_if<std::decay_t<ExPolicy>>(
                        part_begin, part_count, tok, PIKA_FORWARD(F, op),
                        PIKA_FORWARD(Proj, proj));

                    return !tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), PIKA_MOVE(f1),
                    [](std::vector<pika::future<bool>>&& results) {
                        return detail::sequential_find_if_not<
                                   pika::execution::sequenced_policy>(
                                   pika::util::begin(results),
                                   pika::util::end(results),
                                   [](pika::future<bool>& val) {
                                       return val.get();
                                   }) == pika::util::end(results);
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator<FwdIter>::value &&
            traits::is_projected<Proj, FwdIter>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter>
            >::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::none_of is deprecated, use pika::none_of instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        none_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return pika::parallel::v1::detail::none_of().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
            PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // any_of
    namespace detail {
        /// \cond NOINTERNAL
        struct any_of : public detail::algorithm<any_of, bool>
        {
            any_of()
              : any_of::algorithm("any_of")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj>
            static bool sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj)
            {
                return detail::sequential_find_if<ExPolicy>(first, last,
                           util::invoke_projected<F, Proj>(PIKA_FORWARD(F, f),
                               PIKA_FORWARD(Proj, proj))) != last;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& op,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                util::cancellation_token<> tok;
                auto f1 = [op = PIKA_FORWARD(F, op), tok,
                              proj = PIKA_FORWARD(Proj, proj)](
                              FwdIter part_begin,
                              std::size_t part_count) mutable -> bool {
                    detail::sequential_find_if<std::decay_t<ExPolicy>>(
                        part_begin, part_count, tok, PIKA_FORWARD(F, op),
                        PIKA_FORWARD(Proj, proj));

                    return tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), PIKA_MOVE(f1),
                    [](std::vector<pika::future<bool>>&& results) {
                        return detail::sequential_find_if<
                                   pika::execution::sequenced_policy>(
                                   pika::util::begin(results),
                                   pika::util::end(results),
                                   [](pika::future<bool>& val) {
                                       return val.get();
                                   }) != pika::util::end(results);
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator<FwdIter>::value &&
            traits::is_projected<Proj, FwdIter>::value &&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter>
            >::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::any_of is deprecated, use pika::any_of instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        any_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return pika::parallel::v1::detail::any_of().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
            PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // all_of
    namespace detail {
        /// \cond NOINTERNAL
        struct all_of : public detail::algorithm<all_of, bool>
        {
            all_of()
              : all_of::algorithm("all_of")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj>
            static bool sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj)
            {
                return detail::sequential_find_if_not<ExPolicy>(first, last,
                           PIKA_FORWARD(F, f), PIKA_FORWARD(Proj, proj)) == last;
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter first, Sent last, F&& op,
                Proj&& proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        true);
                }

                util::cancellation_token<> tok;
                auto f1 = [op = PIKA_FORWARD(F, op), tok,
                              proj = PIKA_FORWARD(Proj, proj)](
                              FwdIter part_begin,
                              std::size_t part_count) mutable -> bool {
                    detail::sequential_find_if_not<std::decay_t<ExPolicy>>(
                        part_begin, part_count, tok, PIKA_FORWARD(F, op),
                        PIKA_FORWARD(Proj, proj));

                    return !tok.was_cancelled();
                };

                return util::partitioner<ExPolicy, bool>::call(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), PIKA_MOVE(f1),
                    [](std::vector<pika::future<bool>>&& results) {
                        return detail::sequential_find_if_not<
                                   pika::execution::sequenced_policy>(
                                   pika::util::begin(results),
                                   pika::util::end(results),
                                   [](pika::future<bool>& val) {
                                       return val.get();
                                   }) == pika::util::end(results);
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value&&
            pika::traits::is_iterator<FwdIter>::value&&
            traits::is_projected<Proj, FwdIter>::value&&
            traits::is_indirect_callable<ExPolicy, F,
                traits::projected<Proj, FwdIter>
            >::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::all_of is deprecated, use pika::all_of instead")
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        all_of(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj = Proj())
    {
        static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return pika::parallel::v1::detail::all_of().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
            PIKA_FORWARD(Proj, proj));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::none_of
    inline constexpr struct none_of_t final
      : pika::detail::tag_parallel_algorithm<none_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(
            none_of_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::none_of().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename InIter, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            none_of_t, InIter first, InIter last, F&& f)
        {
            static_assert(pika::traits::is_input_iterator<InIter>::value,
                "Required at least input iterator.");

            return pika::parallel::v1::detail::none_of().call(
                pika::execution::seq, first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }
    } none_of{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::any_of
    inline constexpr struct any_of_t final
      : pika::detail::tag_parallel_algorithm<any_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(
            any_of_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::any_of().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename InIter, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            any_of_t, InIter first, InIter last, F&& f)
        {
            static_assert(pika::traits::is_input_iterator<InIter>::value,
                "Required at least input iterator.");

            return pika::parallel::v1::detail::any_of().call(pika::execution::seq,
                first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }
    } any_of{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::all_of
    inline constexpr struct all_of_t final
      : pika::detail::tag_parallel_algorithm<all_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(
            all_of_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::all_of().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename InIter, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(
            all_of_t, InIter first, InIter last, F&& f)
        {
            static_assert(pika::traits::is_input_iterator<InIter>::value,
                "Required at least input iterator.");

            return pika::parallel::v1::detail::all_of().call(pika::execution::seq,
                first, last, PIKA_FORWARD(F, f),
                pika::parallel::util::projection_identity{});
        }
    } all_of{};

}    // namespace pika

#endif    // DOXYGEN
