//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2021 Karame M.Shokooh
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/adjacent_difference.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika {
    ////////////////////////////////////////////////////////////////////////////
    /// Assigns each value in the range given by result its corresponding
    /// element in the range [first, last] and the one preceding it except
    /// *result, which is assigned *first
    ///
    /// \note   Complexity: Exactly (last - first) - 1 application of the
    ///                     binary operator and (last - first) assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     input range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     output range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the sequence of elements
    ///                     the results will be assigned to.
    ///
    /// The difference operations in the parallel \a adjacent_difference invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The difference operations in the parallel \a adjacent_difference invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a adjacent_difference algorithm returns a
    ///           \a pika::future<FwdIter2> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a adjacent_find algorithm returns an iterator to the
    ///           last element in the output range.
    ///
    ///           This overload of \a adjacent_find is available if the user
    ///           decides to provide their algorithm their own binary
    ///           predicate \a op.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    inline typename std::enable_if<pika::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type>::type
    adjacent_difference(
        ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest)

        ///////////////////////////////////////////////////////////////////////////
        /// Assigns each value in the range given by result its corresponding
        /// element in the range [first, last] and the one preceding it except
        /// *result, which is assigned *first
        ///
        /// \note   Complexity: Exactly (last - first) - 1 application of the
        ///                     binary operator and (last - first) assignments.
        ///
        /// \tparam ExPolicy    The type of the execution policy to use (deduced).
        ///                     It describes the manner in which the execution
        ///                     of the algorithm may be parallelized and the manner
        ///                     in which it executes the assignments.
        /// \tparam FwdIter1    The type of the source iterators used for the
        ///                     input range (deduced).
        ///                     This iterator type must meet the requirements of an
        ///                     forward iterator.
        /// \tparam FwdIter2    The type of the source iterators used for the
        ///                     output range (deduced).
        ///                     This iterator type must meet the requirements of an
        ///                     forward iterator.
        /// \tparam Op          The type of the function/function object to use
        ///                     (deduced). Unlike its sequential form, the parallel
        ///                     overload of \a adjacent_difference requires \a Op
        ///                     to meet the requirements of \a CopyConstructible.
        ///
        /// \param policy       The execution policy to use for the scheduling of
        ///                     the iterations.
        /// \param first        Refers to the beginning of the sequence of elements
        ///                     of the range the algorithm will be applied to.
        /// \param last         Refers to the end of the sequence of elements of
        ///                     the range the algorithm will be applied to.
        /// \param dest         Refers to the beginning of the sequence of elements
        ///                     the results will be assigned to.
        /// \param op           The binary operator which returns the difference
        ///                     of elements. The signature should be equivalent
        ///                     to the following:
        ///                     \code
        ///                     bool op(const Type1 &a, const Type1 &b);
        ///                     \endcode \n
        ///                     The signature does not need to have const &, but
        ///                     the function must not modify the objects passed to
        ///                     it. The types \a Type1  must be such
        ///                     that objects of type \a FwdIter1 can be dereferenced
        ///                     and then implicitly converted to the dereferenced
        ///                     type of \a dest.
        ///
        /// The difference operations in the parallel \a adjacent_difference invoked
        /// with an execution policy object of type \a sequenced_policy
        /// execute in sequential order in the calling thread.
        ///
        /// The difference operations in the parallel \a adjacent_difference invoked
        /// with an execution policy object of type \a parallel_policy
        /// or \a parallel_task_policy are permitted to execute in an
        /// unordered fashion in unspecified threads, and indeterminately sequenced
        /// within each thread.
        ///
        /// \returns  The \a adjacent_difference algorithm returns a
        ///           \a pika::future<FwdIter2> if the execution policy is of type
        ///           \a sequenced_task_policy or
        ///           \a parallel_task_policy and
        ///           returns \a FwdIter2 otherwise.
        ///           The \a adjacent_find algorithm returns an iterator to the
        ///           last element in the output range.
        ///
        ///
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Op>
        inline
        typename std::enable_if<pika::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type>::type adjacent_difference(ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, Op&& op)
}    // namespace pika

#else

#include <pika/local/config.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/zip_iterator.hpp>

#include <pika/algorithms/traits/is_value_proxy.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/adjacent_difference.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>
#include <pika/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // adjacent_difference
    namespace detail {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct adjacent_difference
          : public detail::algorithm<adjacent_difference<Iter>, Iter>
        {
            adjacent_difference()
              : adjacent_difference::algorithm("adjacent_difference")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter, typename Op>
            static OutIter sequential(
                ExPolicy, InIter first, Sent last, OutIter dest, Op&& op)
            {
                return sequential_adjacent_difference<ExPolicy>(
                    first, last, dest, PIKA_FORWARD(Op, op));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2, typename Op>
            static typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type
            parallel(ExPolicy&& policy, FwdIter1 first, Sent last,
                FwdIter2 dest, Op&& op)
            {
                using zip_iterator =
                    pika::util::zip_iterator<FwdIter1, FwdIter1, FwdIter2>;
                using result =
                    util::detail::algorithm_result<ExPolicy, FwdIter2>;
                using difference_type =
                    typename std::iterator_traits<FwdIter1>::difference_type;

                if (first == last)
                {
                    return result::get(PIKA_MOVE(dest));
                }

                difference_type count = detail::distance(first, last) - 1;

                FwdIter1 prev = first;
                pika::traits::proxy_value_t<
                    typename std::iterator_traits<FwdIter1>::value_type>
                    tmp = *first++;
                *dest++ = PIKA_MOVE(tmp);

                if (count == 0)
                {
                    return result::get(PIKA_MOVE(dest));
                }

                auto f1 = [op = PIKA_FORWARD(Op, op)](zip_iterator part_begin,
                              std::size_t part_size) mutable {
                    // VS2015RC bails out when op is captured by ref
                    using pika::get;
                    util::loop_n<std::decay_t<ExPolicy>>(
                        part_begin, part_size, [op](auto&& it) mutable {
                            get<2>(*it) =
                                PIKA_INVOKE(op, get<0>(*it), get<1>(*it));
                        });
                };

                auto f2 = [dest, count](
                              std::vector<pika::future<void>>&& data) mutable
                    -> FwdIter2 {
                    // make sure iterators embedded in function object that is
                    // attached to futures are invalidated
                    data.clear();
                    std::advance(dest, count);
                    return dest;
                };

                using pika::util::make_zip_iterator;
                return util::partitioner<ExPolicy, FwdIter2, void>::call(
                    PIKA_FORWARD(ExPolicy, policy),
                    make_zip_iterator(first, prev, dest), count, PIKA_MOVE(f1),
                    PIKA_MOVE(f2));
            }
        };

        /// \endcond
    }    // namespace detail

    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::adjacent_difference is deprecated, use "
        "pika::adjacent_difference instead")
    inline std::enable_if_t<pika::is_execution_policy_v<ExPolicy>,
        util::detail::algorithm_result_t<ExPolicy,
            FwdIter2>> adjacent_difference(ExPolicy&& policy, FwdIter1 first,
        FwdIter1 last, FwdIter2 dest)
    {
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return pika::parallel::v1::detail::adjacent_difference<FwdIter2>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, dest, std::minus<>());
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Op>
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::adjacent_difference is deprecated, use "
        "pika::adjacent_difference instead")
    inline std::enable_if_t<pika::is_execution_policy_v<ExPolicy>,
        util::detail::algorithm_result_t<ExPolicy,
            FwdIter2>> adjacent_difference(ExPolicy&& policy, FwdIter1 first,
        FwdIter1 last, FwdIter2 dest, Op&& op)
    {
        return detail::adjacent_difference<FwdIter2>().call(
            PIKA_FORWARD(ExPolicy, policy), first, last, dest,
            PIKA_FORWARD(Op, op));
    }
}}}    // namespace pika::parallel::v1

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::adjacent_difference
    inline constexpr struct adjacent_difference_t final
      : pika::detail::tag_parallel_algorithm<adjacent_difference_t>
    {
        // clang-format off
        private:
        template <typename FwdIter1, typename FwdIter2,
             PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend FwdIter2 tag_fallback_invoke(pika::adjacent_difference_t,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
                "Required at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::adjacent_difference<FwdIter2>()
                .call(pika::execution::sequenced_policy{}, first, last, dest,
                    std::minus<>());
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
            FwdIter2>
        tag_fallback_invoke(pika::adjacent_difference_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
                "Required at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::adjacent_difference<FwdIter2>()
                .call(PIKA_FORWARD(ExPolicy, policy), first, last, dest,
                    std::minus<>());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename Op,
             PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend FwdIter2 tag_fallback_invoke(pika::adjacent_difference_t,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, Op&& op)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
                "Required at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::adjacent_difference<FwdIter2>()
                .call(pika::execution::sequenced_policy{}, first, last, dest,
                    PIKA_FORWARD(Op, op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename Op,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend pika::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter2>
        tag_fallback_invoke(pika::adjacent_difference_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, Op&& op)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
                "Required at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
                "Required at least forward iterator.");

            return pika::parallel::v1::detail::adjacent_difference<FwdIter2>()
                .call(PIKA_FORWARD(ExPolicy, policy), first, last, dest,
                    PIKA_FORWARD(Op, op));
        }

    } adjacent_difference{};
}    // namespace pika
#endif
