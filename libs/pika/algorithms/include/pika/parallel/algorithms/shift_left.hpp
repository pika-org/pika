//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/shift_left.hpp

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
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
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
    template <typename FwdIter, typename Size>
    FwdIter shift_left(FwdIter first, FwdIter last, Size n);

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
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
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
    template <typename ExPolicy, typename FwdIter, typename Size>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>
    shift_left(ExPolicy&& policy, FwdIter first, FwdIter last, Size n);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/async_local/dataflow.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/modules/execution.hpp>
#include <pika/pack_traversal/unwrap.hpp>

#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/reverse.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/result_types.hpp>
#include <pika/parallel/util/transfer.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // shift_left
    namespace detail {
        template <typename ExPolicy, typename FwdIter, typename Sent>
        pika::future<FwdIter> shift_left_helper(
            ExPolicy policy, FwdIter first, Sent last, FwdIter new_first)
        {
            using non_seq = std::false_type;
            auto p = pika::execution::parallel_task_policy()
                         .on(policy.executor())
                         .with(policy.parameters());

            detail::reverse<FwdIter> r;
            return dataflow(
                [=](pika::future<FwdIter>&& f1) mutable -> pika::future<FwdIter> {
                    f1.get();

                    pika::future<FwdIter> f = r.call2(p, non_seq(), first, last);
                    return f.then(
                        [=](pika::future<FwdIter>&& f) mutable -> FwdIter {
                            f.get();
                            std::advance(
                                first, detail::distance(new_first, last));
                            return first;
                        });
                },
                r.call2(p, non_seq(), new_first, last));
        }

        /* Sequential shift_left implementation inspired
        from https://github.com/danra/shift_proposal */

        template <typename FwdIter, typename Sent, typename Size>
        static constexpr FwdIter sequential_shift_left(
            FwdIter first, Sent last, Size n, std::size_t dist)
        {
            auto mid = std::next(first, n);

            if constexpr (pika::traits::is_random_access_iterator_v<FwdIter>)
            {
                return parallel::util::get_second_element(
                    util::move_n(mid, dist - n, PIKA_MOVE(first)));
            }
            else
            {
                return parallel::util::get_second_element(
                    util::move(PIKA_MOVE(mid), PIKA_MOVE(last), PIKA_MOVE(first)));
            }
        }

        template <typename FwdIter2>
        struct shift_left
          : public detail::algorithm<shift_left<FwdIter2>, FwdIter2>
        {
            shift_left()
              : shift_left::algorithm("shift_left")
            {
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename Size>
            static FwdIter sequential(
                ExPolicy, FwdIter first, Sent last, Size n)
            {
                auto dist =
                    static_cast<std::size_t>(detail::distance(first, last));
                if (n <= 0 || static_cast<std::size_t>(n) >= dist)
                {
                    return first;
                }

                return detail::sequential_shift_left(first, last, n, dist);
            }

            template <typename ExPolicy, typename Sent, typename Size>
            static typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type
            parallel(ExPolicy&& policy, FwdIter2 first, Sent last, Size n)
            {
                auto dist =
                    static_cast<std::size_t>(detail::distance(first, last));
                if (n <= 0 || static_cast<std::size_t>(n) >= dist)
                {
                    return parallel::util::detail::algorithm_result<ExPolicy,
                        FwdIter2>::get(PIKA_MOVE(first));
                }

                return util::detail::algorithm_result<ExPolicy, FwdIter2>::get(
                    shift_left_helper(
                        policy, first, last, std::next(first, n)));
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::shift_left
    inline constexpr struct shift_left_t final
      : pika::functional::detail::tag_fallback<shift_left_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value)>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            shift_left_t, FwdIter first, FwdIter last, Size n)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_left<FwdIter>().call(
                pika::execution::seq, first, last, n);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value)>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(shift_left_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, Size n)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_left<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, n);
        }
    } shift_left{};
}    // namespace pika

#endif    // DOXYGEN
