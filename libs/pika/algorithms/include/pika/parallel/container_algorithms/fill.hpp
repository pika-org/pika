//  Copyright (c) 2018 Christopher Ogle
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/fill.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    /// Assigns the given value to the elements in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The comparisons in the parallel \a fill algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill algorithm returns a \a pika::future<void> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename Rng, typename T>
    typename util::detail::algorithm_result<ExPolicy>::type
    fill(ExPolicy&& policy, Rng&& rng, T const& value);

    /// Assigns the given value value to the first count elements in the range
    /// beginning at first if count > 0. Does nothing otherwise.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, for
    ///         count > 0.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iterator    The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The comparisons in the parallel \a fill_n algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill_n algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill_n algorithm returns a \a pika::future<void> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename Iterator, typename Size, typename T>
    typename util::detail::algorithm_result<ExPolicy, Iterator>::type
    fill_n(ExPolicy&& policy, Iterator first, Size count, T const& value);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_range.hpp>

#include <pika/parallel/algorithms/fill.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename Rng, typename T,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_range<Rng>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::fill is deprecated, use pika::ranges::fill instead")
        typename util::detail::algorithm_result<ExPolicy,
            typename pika::traits::range_traits<Rng>::iterator_type>::type
        fill(ExPolicy&& policy, Rng&& rng, T value)
    {
        using iterator_type =
            typename pika::traits::range_traits<Rng>::iterator_type;

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(pika::traits::is_forward_iterator<iterator_type>::value,
            "Requires at least forward iterator.");

        return detail::fill<iterator_type>().call(PIKA_FORWARD(ExPolicy, policy),
            pika::util::begin(rng), pika::util::end(rng), value);
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename Size, typename T,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_range<Rng>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::fill_n is deprecated, use pika::ranges::fill_n instead")
        typename util::detail::algorithm_result<ExPolicy,
            typename pika::traits::range_traits<Rng>::iterator_type>::type
        fill_n(ExPolicy&& policy, Rng& rng, Size count, T value)
    {
        using iterator_type =
            typename pika::traits::range_traits<Rng>::iterator_type;

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(pika::traits::is_forward_iterator<iterator_type>::value,
            "Requires at least forward iterator.");

        return detail::fill_n<iterator_type>().call(
            PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng), count, value);
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

}}}    // namespace pika::parallel::v1

namespace pika { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::ranges::fill
    inline constexpr struct fill_t final
      : pika::detail::tag_parallel_algorithm<fill_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename T = typename std::iterator_traits<
                pika::traits::range_iterator_t<Rng>>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            typename pika::traits::range_traits<Rng>::iterator_type>::type
        tag_fallback_invoke(
            fill_t, ExPolicy&& policy, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                pika::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::fill<iterator_type>().call(
                PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
                pika::util::end(rng), value);
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename T = typename std::iterator_traits<Iter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_sentinel_for<Sent, Iter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_invoke(
            fill_t, ExPolicy&& policy, Iter first, Sent last, T const& value)
        {
            static_assert(pika::traits::is_forward_iterator<Iter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::fill<Iter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, value);
        }

        // clang-format off
        template <typename Rng,
            typename T = typename std::iterator_traits<
                pika::traits::range_iterator_t<Rng>>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend pika::traits::range_iterator_t<Rng> tag_fallback_invoke(
            fill_t, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                pika::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::fill<iterator_type>().call(
                pika::execution::seq, pika::util::begin(rng), pika::util::end(rng),
                value);
        }

        // clang-format off
        template <typename Iter, typename Sent,
            typename T = typename std::iterator_traits<Iter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_sentinel_for<Sent, Iter>::value
            )>
        // clang-format on
        friend Iter tag_fallback_invoke(
            fill_t, Iter first, Sent last, T const& value)
        {
            static_assert(pika::traits::is_forward_iterator<Iter>::value,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::fill<Iter>().call(
                pika::execution::seq, first, last, value);
        }
    } fill{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::ranges::fill_n
    inline constexpr struct fill_n_t final
      : pika::detail::tag_parallel_algorithm<fill_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename T = typename std::iterator_traits<
                pika::traits::range_iterator_t<Rng>>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            pika::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(
            fill_n_t, ExPolicy&& policy, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (pika::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (pika::parallel::v1::detail::is_negative(pika::util::size(rng)))
            {
                auto first = pika::util::begin(rng);
                return pika::parallel::util::detail::algorithm_result<ExPolicy,
                    iterator_type>::get(PIKA_MOVE(first));
            }

            return pika::parallel::v1::detail::fill_n<iterator_type>().call(
                PIKA_FORWARD(ExPolicy, policy), pika::util::begin(rng),
                pika::util::size(rng), value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            typename T = typename std::iterator_traits<FwdIter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(fill_n_t, ExPolicy&& policy, FwdIter first,
            Size count, T const& value)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (pika::parallel::v1::detail::is_negative(count))
            {
                return pika::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(PIKA_MOVE(first));
            }

            return pika::parallel::v1::detail::fill_n<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, std::size_t(count),
                value);
        }

        // clang-format off
        template <typename Rng,
            typename T = typename std::iterator_traits<
                pika::traits::range_iterator_t<Rng>>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename pika::traits::range_traits<Rng>::iterator_type
        tag_fallback_invoke(fill_n_t, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename pika::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (pika::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (pika::parallel::v1::detail::is_negative(pika::util::size(rng)))
            {
                return pika::util::begin(rng);
            }

            return pika::parallel::v1::detail::fill_n<iterator_type>().call(
                pika::execution::seq, pika::util::begin(rng),
                pika::util::size(rng), value);
        }

        // clang-format off
        template <typename FwdIter, typename Size,
            typename T = typename std::iterator_traits<FwdIter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            fill_n_t, FwdIter first, Size count, T const& value)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (pika::parallel::v1::detail::is_negative(count))
            {
                return first;
            }

            return pika::parallel::v1::detail::fill_n<FwdIter>().call(
                pika::execution::seq, first, std::size_t(count), value);
        }
    } fill_n{};

}}    // namespace pika::ranges

#endif    // DOXYGEN
