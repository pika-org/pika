//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/shift_right.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the end of the range. For every integer i in [0, last - first - n),
    /// moves the element originally at position first + i to position first
    /// + n + i.
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
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable.
    ///
    /// \returns  The \a shift_right algorithm returns \a FwdIter.
    ///           The \a shift_right algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename FwdIter, typename Size>
    FwdIter shift_right(FwdIter first, FwdIter last, Size n);

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the end of the range. For every integer i in [0, last - first - n),
    /// moves the element originally at position first + i to position first
    /// + n + i.
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
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable.
    ///
    /// \returns  The \a shift_right algorithm returns a
    ///           \a pika::future<FwdIter> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a shift_right algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>
    shift_right(ExPolicy&& policy, FwdIter first, FwdIter last, Size n);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/async_local/dataflow.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/modules/execution.hpp>
#include <pika/pack_traversal/unwrap.hpp>

#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/copy.hpp>
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
    // shift_right
    namespace detail {
        template <typename ExPolicy, typename FwdIter, typename Sent>
        pika::future<FwdIter> shift_right_helper(
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
                            return new_first;
                        });
                },
                r.call2(p, non_seq(), first, new_first));
        }

        /* Sequential shift_right implementation borrowed
        from https://github.com/danra/shift_proposal */

        template <class I>
        using difference_type_t =
            typename std::iterator_traits<I>::difference_type;

        template <class I>
        using iterator_category_t =
            typename std::iterator_traits<I>::iterator_category;

        template <class I, class Tag, class = void>
        constexpr bool is_category = false;
        template <class I, class Tag>
        constexpr bool is_category<I, Tag,
            std::enable_if_t<
                std::is_convertible_v<iterator_category_t<I>, Tag>>> = true;

        template <typename FwdIter>
        FwdIter sequential_shift_right(FwdIter first, FwdIter last,
            difference_type_t<FwdIter> n, std::size_t dist)
        {
            if constexpr (is_category<FwdIter, std::bidirectional_iterator_tag>)
            {
                auto mid = std::next(first, dist - n);
                return std::move_backward(
                    PIKA_MOVE(first), PIKA_MOVE(mid), PIKA_MOVE(last));
            }
            else
            {
                auto result = std::next(first, n);
                auto lead = result;
                auto trail = first;

                for (; trail != result; ++lead, void(++trail))
                {
                    if (lead == last)
                    {
                        std::move(PIKA_MOVE(first), PIKA_MOVE(trail), result);
                        return result;
                    }
                }

                for (;;)
                {
                    for (auto mid = first; mid != result;
                         ++lead, void(++trail), ++mid)
                    {
                        if (lead == last)
                        {
                            trail = std::move(mid, result, PIKA_MOVE(trail));
                            std::move(PIKA_MOVE(first), PIKA_MOVE(mid),
                                PIKA_MOVE(trail));
                            return result;
                        }
                        std::iter_swap(mid, trail);
                    }
                }
            }
        }

        template <typename FwdIter2>
        struct shift_right
          : public detail::algorithm<shift_right<FwdIter2>, FwdIter2>
        {
            shift_right()
              : shift_right::algorithm("shift_right")
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

                auto last_iter = detail::advance_to_sentinel(first, last);
                return detail::sequential_shift_right(
                    first, last_iter, difference_type_t<FwdIter>(n), dist);
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

                auto new_first = std::next(first, dist - n);
                return util::detail::algorithm_result<ExPolicy, FwdIter2>::get(
                    shift_right_helper(policy, first, last, new_first));
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::shift_right
    inline constexpr struct shift_right_t final
      : pika::functional::detail::tag_fallback<shift_right_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value)>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            shift_right_t, FwdIter first, FwdIter last, Size n)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_right<FwdIter>().call(
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
        tag_fallback_invoke(shift_right_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, Size n)
        {
            static_assert(pika::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::shift_right<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, n);
        }
    } shift_right{};
}    // namespace pika

#endif    // DOXYGEN
