//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/destroy.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first operations.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
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
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a destroy algorithm returns a
    ///           \a pika::future<void>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename FwdIter>
    typename util::detail::algorithm_result<ExPolicy>::type
    destroy(ExPolicy&& policy, FwdIter first, FwdIter last);

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, first + count).
    ///
    /// \note   Complexity: Performs exactly \a count operations, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply this algorithm to.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The operations in the parallel \a destroy_n
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The operations in the parallel \a destroy_n
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a destroy_n algorithm returns a
    ///           \a pika::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a destroy_n algorithm returns the
    ///           iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    destroy_n(ExPolicy&& policy, FwdIter first, Size count);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>

#include <pika/execution/algorithms/detail/is_negative.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/foreach_partitioner.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // destroy
    namespace detail {
        /// \cond NOINTERNAL

        // provide our own implementation of std::destroy
        // as some versions of MSVC horribly fail at compiling it for some types
        // T
        template <typename Iter, typename Sent>
        Iter sequential_destroy(Iter first, Sent last)
        {
            using value_type = typename std::iterator_traits<Iter>::value_type;

            for (/* */; first != last; ++first)
            {
                std::addressof(*first)->~value_type();
            }
            return first;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter>
        typename util::detail::algorithm_result<ExPolicy, Iter>::type
        parallel_sequential_destroy_n(
            ExPolicy&& policy, Iter first, std::size_t count)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy, Iter>::get(
                    PIKA_MOVE(first));
            }

            return util::foreach_partitioner<ExPolicy>::call(
                PIKA_FORWARD(ExPolicy, policy), first, count,
                [](Iter first, std::size_t count, std::size_t) {
                    return util::loop_n<std::decay_t<ExPolicy>>(
                        first, count, [](Iter it) -> void {
                            using value_type =
                                typename std::iterator_traits<Iter>::value_type;

                            std::addressof(*it)->~value_type();
                        });
                },
                util::projection_identity());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter>
        struct destroy : public detail::algorithm<destroy<FwdIter>, FwdIter>
        {
            destroy()
              : destroy::algorithm("destroy")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent>
            static Iter sequential(ExPolicy, Iter first, Sent last)
            {
                return sequential_destroy(first, last);
            }

            template <typename ExPolicy, typename Iter, typename Sent>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, Sent last)
            {
                return parallel_sequential_destroy_n(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last));
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator<FwdIter>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(
        0, 1, "pika::parallel::destroy is deprecated, use pika::destroy instead")
        typename util::detail::algorithm_result<ExPolicy>::type
        destroy(ExPolicy&& policy, FwdIter first, FwdIter last)
    {
        static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::destroy<FwdIter>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last);
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // destroy_n
    namespace detail {
        /// \cond NOINTERNAL

        // provide our own implementation of std::destroy
        // as some versions of MSVC horribly fail at compiling it for some
        // types T
        template <typename Iter>
        Iter sequential_destroy_n(Iter first, std::size_t count)
        {
            using value_type = typename std::iterator_traits<Iter>::value_type;

            for (/* */; count != 0; (void) ++first, --count)
            {
                std::addressof(*first)->~value_type();
            }

            return first;
        }

        template <typename FwdIter>
        struct destroy_n : public detail::algorithm<destroy_n<FwdIter>, FwdIter>
        {
            destroy_n()
              : destroy_n::algorithm("destroy_n")
            {
            }

            template <typename ExPolicy, typename Iter>
            static Iter sequential(ExPolicy, Iter first, std::size_t count)
            {
                return sequential_destroy_n(first, count);
            }

            template <typename ExPolicy, typename Iter>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, std::size_t count)
            {
                return parallel_sequential_destroy_n(
                    PIKA_FORWARD(ExPolicy, policy), first, count);
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename Size,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator<FwdIter>::value
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::destroy_n is deprecated, use pika::destroy_n instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        destroy_n(ExPolicy&& policy, FwdIter first, Size count)
    {
        static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                PIKA_MOVE(first));
        }

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::destroy_n<FwdIter>().call(
            PIKA_FORWARD(ExPolicy, policy), first, std::size_t(count));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::destroy
    inline constexpr struct destroy_t final
      : pika::detail::tag_parallel_algorithm<destroy_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<
            ExPolicy>::type
        tag_fallback_invoke(
            destroy_t, ExPolicy&& policy, FwdIter first, FwdIter last)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            return pika::parallel::util::detail::algorithm_result<ExPolicy>::get(
                pika::parallel::v1::detail::destroy<FwdIter>().call(
                    PIKA_FORWARD(ExPolicy, policy), first, last));
        }

        // clang-format off
        template <typename FwdIter,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend void tag_fallback_invoke(destroy_t, FwdIter first, FwdIter last)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Required at least forward iterator.");

            pika::parallel::v1::detail::destroy<FwdIter>().call(
                pika::execution::seq, first, last);
        }
    } destroy{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::destroy_n
    inline constexpr struct destroy_n_t final
      : pika::detail::tag_parallel_algorithm<destroy_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(
            destroy_n_t, ExPolicy&& policy, FwdIter first, Size count)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (pika::parallel::v1::detail::is_negative(count))
            {
                return pika::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(PIKA_MOVE(first));
            }

            return pika::parallel::v1::detail::destroy_n<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, std::size_t(count));
        }

        // clang-format off
        template <typename FwdIter, typename Size,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            destroy_n_t, FwdIter first, Size count)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (pika::parallel::v1::detail::is_negative(count))
            {
                return first;
            }

            return pika::parallel::v1::detail::destroy_n<FwdIter>().call(
                pika::execution::seq, first, std::size_t(count));
        }
    } destroy_n{};
}    // namespace pika

#endif    // DOXYGEN
