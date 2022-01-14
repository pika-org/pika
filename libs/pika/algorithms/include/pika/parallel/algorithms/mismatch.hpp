//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/mismatch.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    /// Returns true if the range [first1, last1) is mismatch to the range
    /// [first2, last2), and false otherwise.
    ///
    /// \note   Complexity: At most min(last1 - first1, last2 - first2)
    ///         applications of the predicate \a f. If \a FwdIter1
    ///         and \a FwdIter2 meet the requirements of \a RandomAccessIterator
    ///         and (last1 - first1) != (last2 - first2) then no applications
    ///         of the predicate \a f are made.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a mismatch requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
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
    ///                     elements should be treated as mismatch. The signature
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
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     The two ranges are considered mismatch if, for every iterator
    ///           i in the range [first1,last1), *i mismatchs *(first2 + (i - first1)).
    ///           This overload of mismatch uses operator== to determine if two
    ///           elements are mismatch.
    ///
    /// \returns  The \a mismatch algorithm returns a \a pika::future<bool> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a mismatch algorithm returns true if the elements in the
    ///           two ranges are mismatch, otherwise it returns false.
    ///           If the length of the range [first1, last1) does not mismatch
    ///           the length of the range [first2, last2), it returns false.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    util::detail::algorithm_result_t<ExPolicy, std::pair<FwdIter1, FwdIter2>>
    mismatch(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred());

    /// Returns std::pair with iterators to the first two non-equivalent
    /// elements.
    ///
    /// \note   Complexity: At most \a last1 - \a first1 applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a mismatch requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as mismatch. The signature
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
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a mismatch algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a mismatch algorithm returns a
    ///           \a pika::future<std::pair<FwdIter1, FwdIter2> > if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a std::pair<FwdIter1, FwdIter2> otherwise.
    ///           The \a mismatch algorithm returns the first mismatching pair
    ///           of elements from two ranges: one defined by [first1, last1)
    ///           and another defined by [first2, last2).
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    util::detail::algorithm_result_t<ExPolicy, std::pair<FwdIter1, FwdIter2>>
    mismatch(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2,
        Pred&& op = Pred());

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>

#include <pika/execution/algorithms/detail/predicates.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>
#include <pika/parallel/util/result_types.hpp>
#include <pika/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // mismatch (binary)
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename F, typename Proj1, typename Proj2>
        constexpr util::in_in_result<Iter1, Iter2> sequential_mismatch_binary(
            Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2, F&& f,
            Proj1&& proj1, Proj2&& proj2)
        {
            while (first1 != last1 && first2 != last2 &&
                PIKA_INVOKE(
                    f, PIKA_INVOKE(proj1, *first1), PIKA_INVOKE(proj2, *first2)))
            {
                (void) ++first1, ++first2;
            }
            return {first1, first2};
        }

        template <typename IterPair>
        struct mismatch_binary
          : public detail::algorithm<mismatch_binary<IterPair>, IterPair>
        {
            mismatch_binary()
              : mismatch_binary::algorithm("mismatch_binary")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static constexpr util::in_in_result<Iter1, Iter2> sequential(
                ExPolicy, Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2,
                F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                return sequential_mismatch_binary(first1, last1, first2, last2,
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Proj1, proj1),
                    PIKA_FORWARD(Proj2, proj2));
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename F, typename Proj1,
                typename Proj2>
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_in_result<Iter1, Iter2>>
            parallel(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
                Sent2 last2, F&& f, Proj1&& proj1, Proj2&& proj2)
            {
                if (first1 == last1 || first2 == last2)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        util::in_in_result<Iter1, Iter2>>::
                        get(util::in_in_result<Iter1, Iter2>{first1, first2});
                }

                using difference_type1 =
                    typename std::iterator_traits<Iter1>::difference_type;
                difference_type1 count1 = detail::distance(first1, last1);

                // The specification of std::mismatch(_binary) states that if FwdIter1
                // and FwdIter2 meet the requirements of RandomAccessIterator and
                // last1 - first1 != last2 - first2 then no applications of the
                // predicate p are made.
                //
                // We perform this check for any iterator type better than input
                // iterators. This could turn into a QoI issue.
                using difference_type2 =
                    typename std::iterator_traits<Iter2>::difference_type;
                difference_type2 count2 = detail::distance(first2, last2);
                if (count1 != count2)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        util::in_in_result<Iter1, Iter2>>::
                        get(util::in_in_result<Iter1, Iter2>{first1, first2});
                }

                using zip_iterator = pika::util::zip_iterator<Iter1, Iter2>;
                using reference = typename zip_iterator::reference;

                util::cancellation_token<std::size_t> tok(count1);

                // Note: replacing the invoke() with PIKA_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, f = PIKA_FORWARD(F, f),
                              proj1 = PIKA_FORWARD(Proj1, proj1),
                              proj2 = PIKA_FORWARD(Proj2, proj2)](
                              zip_iterator it, std::size_t part_count,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, it,
                        part_count, tok,
                        [&f, &proj1, &proj2, &tok](
                            reference t, std::size_t i) mutable -> void {
                            if (!pika::util::invoke(f,
                                    pika::util::invoke(proj1, pika::get<0>(t)),
                                    pika::util::invoke(proj2, pika::get<1>(t))))
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 = [=](std::vector<pika::future<void>>&& data) mutable
                    -> util::in_in_result<Iter1, Iter2> {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    data.clear();
                    difference_type1 mismatched =
                        static_cast<difference_type1>(tok.get_data());
                    if (mismatched != count1)
                    {
                        std::advance(first1, mismatched);
                        std::advance(first2, mismatched);
                    }
                    else
                    {
                        first1 = detail::advance_to_sentinel(first1, last1);
                        first2 = detail::advance_to_sentinel(first2, last2);
                    }
                    return {first1, first2};
                };

                return util::partitioner<ExPolicy,
                    util::in_in_result<Iter1, Iter2>,
                    void>::call_with_index(PIKA_FORWARD(ExPolicy, policy),
                    pika::util::make_zip_iterator(first1, first2), count1, 1,
                    PIKA_MOVE(f1), PIKA_MOVE(f2));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename I1, typename I2>
        std::pair<I1, I2> get_pair(util::in_in_result<I1, I2>&& p)
        {
            return {p.in1, p.in2};
        }

        template <typename I1, typename I2>
        pika::future<std::pair<I1, I2>> get_pair(
            pika::future<util::in_in_result<I1, I2>>&& f)
        {
            return pika::make_future<std::pair<I1, I2>>(PIKA_MOVE(f),
                [](util::in_in_result<I1, I2>&& p) -> std::pair<I1, I2> {
                    return {PIKA_MOVE(p.in1), PIKA_MOVE(p.in2)};
                });
        }
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy> &&
            pika::traits::is_iterator_v<FwdIter1> &&
            pika::traits::is_iterator_v<FwdIter2> &&
            pika::is_invocable_v<Pred,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter2>::value_type
            >
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::mismatch is deprecated, use pika::mismatch instead")
        util::detail::algorithm_result_t<ExPolicy,
            std::pair<FwdIter1, FwdIter2>> mismatch(ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred&& op = Pred())
    {
        static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
            "Requires at least forward iterator.");
        static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
            "Requires at least forward iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::get_pair(
            detail::mismatch_binary<util::in_in_result<FwdIter1, FwdIter2>>()
                .call(PIKA_FORWARD(ExPolicy, policy), first1, last1, first2,
                    last2, PIKA_FORWARD(Pred, op)));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // mismatch
    namespace detail {

        template <typename IterPair>
        struct mismatch : public detail::algorithm<mismatch<IterPair>, IterPair>
        {
            mismatch()
              : mismatch::algorithm("mismatch")
            {
            }

            template <typename ExPolicy, typename InIter1, typename Sent,
                typename InIter2, typename F>
            static constexpr IterPair sequential(
                ExPolicy, InIter1 first1, Sent last1, InIter2 first2, F&& f)
            {
                while (first1 != last1 && PIKA_INVOKE(f, *first1, *first2))
                {
                    ++first1, ++first2;
                }
                return std::make_pair(first1, first2);
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2, typename F>
            static util::detail::algorithm_result_t<ExPolicy, IterPair>
            parallel(ExPolicy&& policy, FwdIter1 first1, Sent last1,
                FwdIter2 first2, F&& f)
            {
                if (first1 == last1)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        IterPair>::get(std::make_pair(first1, first2));
                }

                using difference_type =
                    typename std::iterator_traits<FwdIter1>::difference_type;
                difference_type count = detail::distance(first1, last1);

                using zip_iterator =
                    pika::util::zip_iterator<FwdIter1, FwdIter2>;
                using reference = typename zip_iterator::reference;

                util::cancellation_token<std::size_t> tok(count);

                // Note: replacing the invoke() with PIKA_INVOKE()
                // below makes gcc generate errors
                auto f1 = [tok, f = PIKA_FORWARD(F, f)](zip_iterator it,
                              std::size_t part_count,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n<std::decay_t<ExPolicy>>(base_idx, it,
                        part_count, tok,
                        [&f, &tok](reference t, std::size_t i) mutable -> void {
                            if (!pika::util::invoke(
                                    f, pika::get<0>(t), pika::get<1>(t)))
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 = [=](std::vector<pika::future<void>>&& data) mutable
                    -> std::pair<FwdIter1, FwdIter2> {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    data.clear();
                    difference_type mismatched =
                        static_cast<difference_type>(tok.get_data());
                    if (mismatched != count)
                        std::advance(first1, mismatched);
                    else
                        first1 = detail::advance_to_sentinel(first1, last1);

                    std::advance(first2, mismatched);
                    return std::make_pair(first1, first2);
                };

                return util::partitioner<ExPolicy, IterPair,
                    void>::call_with_index(PIKA_FORWARD(ExPolicy, policy),
                    pika::util::make_zip_iterator(first1, first2), count, 1,
                    PIKA_MOVE(f1), PIKA_MOVE(f2));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy_v<ExPolicy> &&
            pika::traits::is_iterator_v<FwdIter1> &&
            pika::traits::is_iterator_v<FwdIter2> &&
            pika::is_invocable_v<Pred,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter2>::value_type
            >
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::mismatch is deprecated, use pika::mismatch instead")
        util::detail::algorithm_result_t<ExPolicy,
            std::pair<FwdIter1, FwdIter2>> mismatch(ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2,
            Pred&& op = Pred())
    {
        static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
            "Requires at least forward iterator.");
        static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
            "Requires at least forward iterator.");

        using result_type = std::pair<FwdIter1, FwdIter2>;

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::mismatch<result_type>().call(
            PIKA_FORWARD(ExPolicy, policy), first1, last1, first2,
            PIKA_FORWARD(Pred, op));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

}}}    // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::mismatch
    inline constexpr struct mismatch_t final
      : pika::detail::tag_parallel_algorithm<mismatch_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2> &&
                pika::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>
        tag_fallback_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred&& op)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::get_pair(
                pika::parallel::v1::detail::mismatch_binary<
                    pika::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(PIKA_FORWARD(ExPolicy, policy), first1, last1, first2,
                        last2, PIKA_FORWARD(Pred, op),
                        pika::parallel::util::projection_identity{},
                        pika::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>
        tag_fallback_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::get_pair(
                pika::parallel::v1::detail::mismatch_binary<
                    pika::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(PIKA_FORWARD(ExPolicy, policy), first1, last1, first2,
                        last2, pika::parallel::v1::detail::equal_to{},
                        pika::parallel::util::projection_identity{},
                        pika::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2> &&
                pika::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>
        tag_fallback_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, Pred&& op)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(PIKA_FORWARD(ExPolicy, policy), first1, last1, first2,
                    PIKA_FORWARD(Pred, op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            std::pair<FwdIter1, FwdIter2>>
        tag_fallback_invoke(mismatch_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(PIKA_FORWARD(ExPolicy, policy), first1, last1, first2,
                    pika::parallel::v1::detail::equal_to{});
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename Pred,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2> &&
                pika::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_fallback_invoke(mismatch_t,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred&& op)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::get_pair(
                pika::parallel::v1::detail::mismatch_binary<
                    pika::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(pika::execution::seq, first1, last1, first2, last2,
                        PIKA_FORWARD(Pred, op),
                        pika::parallel::util::projection_identity{},
                        pika::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_fallback_invoke(mismatch_t,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::get_pair(
                pika::parallel::v1::detail::mismatch_binary<
                    pika::parallel::util::in_in_result<FwdIter1, FwdIter2>>()
                    .call(pika::execution::seq, first1, last1, first2, last2,
                        pika::parallel::v1::detail::equal_to{},
                        pika::parallel::util::projection_identity{},
                        pika::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename Pred,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2> &&
                pika::is_invocable_v<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_fallback_invoke(mismatch_t,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, Pred&& op)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(pika::execution::seq, first1, last1, first2,
                    PIKA_FORWARD(Pred, op));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend std::pair<FwdIter1, FwdIter2> tag_fallback_invoke(
            mismatch_t, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::mismatch<
                std::pair<FwdIter1, FwdIter2>>()
                .call(pika::execution::seq, first1, last1, first2,
                    pika::parallel::v1::detail::equal_to{});
        }

    } mismatch{};
}    // namespace pika

#endif    // DOXYGEN
