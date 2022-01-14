//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2021 Akhli J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/swap_ranges.hpp

#if defined(DOXYGEN)

namespace pika {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a
    ///  last1
    ///
    /// \tparam FwdIter1    The type of the first range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the second range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second  sequence of
    ///                     elements the algorithm will be applied to.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked without an execution policy object  execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns \a FwdIter2.
    ///           The \a swap_ranges algorithm returns iterator to the element
    ///           past the last element exchanged in the range beginning with
    ///           \a first2.
    ///
    template <typename FwdIter1, typename FwdIter2>
    FwdIter2 swap_ranges(FwdIter1 first1, FwdIter1 last1, FwdIter2 first2);

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a last1
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the swap operations.
    /// \tparam FwdIter1    The type of the first range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the second range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second  sequence of
    ///                     elements the algorithm will be applied to.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns a
    ///           \a pika::future<FwdIter2>  if the execution policy is of
    ///           type \a parallel_task_policy and returns \a FwdIter2
    ///           otherwise.
    ///           The \a swap_ranges algorithm returns iterator to the element
    ///           past the last element exchanged in the range beginning with
    ///           \a first2.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    swap_ranges(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#pragma once

#include <pika/local/config.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>

#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/algorithms/for_each.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/projection_identity.hpp>
#include <pika/parallel/util/result_types.hpp>
#include <pika/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { inline namespace v1 {
    template <typename Iter1, typename Iter2>
    using swap_ranges_result = pika::parallel::util::in_in_result<Iter1, Iter2>;

    ///////////////////////////////////////////////////////////////////////////
    // swap ranges
    namespace detail {
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Size>
        typename util::detail::algorithm_result<ExPolicy,
            swap_ranges_result<FwdIter1, FwdIter2>>::type
        parallel_swap_ranges(
            ExPolicy&& policy, FwdIter1 first1, FwdIter2 first2, Size n)
        {
            using zip_iterator = pika::util::zip_iterator<FwdIter1, FwdIter2>;
            using reference = typename zip_iterator::reference;

            return get_iter_in_in_result(for_each_n<zip_iterator>().call(
                PIKA_FORWARD(ExPolicy, policy),
                pika::util::make_zip_iterator(first1, first2), n,
                [](reference t) -> void {
                    using pika::get;
                    std::swap(get<0>(t), get<1>(t));
                },
                util::projection_identity()));
        }

        template <typename IterPair>
        struct swap_ranges
          : public detail::algorithm<swap_ranges<IterPair>, IterPair>
        {
            swap_ranges()
              : swap_ranges::algorithm("swap_ranges")
            {
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2>
            static constexpr FwdIter2 sequential(
                ExPolicy, FwdIter1 first1, Sent last1, FwdIter2 first2)
            {
                while (first1 != last1)
                {
#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                    std::ranges::iter_swap(first1++, first2++);
#else
                    std::iter_swap(first1++, first2++);
#endif
                }
                return first2;
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2>
            static constexpr swap_ranges_result<FwdIter1, FwdIter2> sequential(
                ExPolicy, FwdIter1 first1, Sent1 last1, FwdIter2 first2,
                Sent2 last2)
            {
                while (first1 != last1 && first2 != last2)
                {
#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                    std::ranges::iter_swap(first1++, first2++);
#else
                    std::iter_swap(first1++, first2++);
#endif
                }
                return swap_ranges_result<FwdIter1, FwdIter2>{first1, first2};
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type
            parallel(
                ExPolicy&& policy, FwdIter1 first1, Sent last1, FwdIter2 first2)
            {
                return util::get_in2_element(
                    parallel_swap_ranges(PIKA_FORWARD(ExPolicy, policy), first1,
                        first2, detail::distance(first1, last1)));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2>
            static typename util::detail::algorithm_result<ExPolicy,
                swap_ranges_result<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2)
            {
                auto dist1 = detail::distance(first1, last1);
                auto dist2 = detail::distance(first2, last2);
                return parallel_swap_ranges(PIKA_FORWARD(ExPolicy, policy),
                    first1, first2, dist1 < dist2 ? dist1 : dist2);
            }
        };
        /// \endcond
    }    // namespace detail

    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::transform_exclusive_scan is deprecated, use "
        "pika::transform_exclusive_scan instead")
    inline typename std::enable_if<pika::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type>::type
        swap_ranges(
            ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
    {
        static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
            "Requires at least forward iterator.");
        static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
            "Requires at least forward iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::swap_ranges<FwdIter2>().call(
            PIKA_FORWARD(ExPolicy, policy), first1, last1, first2);
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::swap_ranges
    inline constexpr struct swap_ranges_t final
      : pika::detail::tag_parallel_algorithm<swap_ranges_t>
    {
        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend FwdIter2 tag_fallback_invoke(pika::swap_ranges_t, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::swap_ranges<FwdIter2>().call(
                pika::execution::seq, first1, last1, first2);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_invoke(pika::swap_ranges_t, ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            return pika::parallel::v1::detail::swap_ranges<FwdIter2>().call(
                PIKA_FORWARD(ExPolicy, policy), first1, last1, first2);
        }
    } swap_ranges{};
}    // namespace pika

#endif    // DOXYGEN
