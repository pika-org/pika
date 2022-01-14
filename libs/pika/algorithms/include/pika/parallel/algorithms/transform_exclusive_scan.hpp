//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_exclusive_scan.hpp

#pragma once

#if defined(DOXYGEN)

namespace pika {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, conv(*first), ...,
    /// conv(*(first + (i - result) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a op and \a conv.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Conv        The type of the unary function object used for
    ///                     the conversion operation.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param conv         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     R fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a transform_exclusive_scan algorithm returns a
    ///           returns \a OutIter.
    ///           The \a transform_exclusive_scan algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or subranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_exclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename InIter, typename OutIter, typename T, typename BinOp,
        typename UnOp>
    OutIter transform_exclusive_scan(InIter first, InIter last, OutIter dest,
        T init, BinOp&& binary_op, UnOp&& unary_op);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, conv(*first), ...,
    /// conv(*(first + (i - result) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a op and \a conv.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Conv        The type of the unary function object used for
    ///                     the conversion operation.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param conv         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     R fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a transform_exclusive_scan algorithm returns a
    ///           \a pika::future<FwdIter2> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a transform_exclusive_scan algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or subranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_exclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename BinOp, typename UnOp>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        FwdIter2>::type
     transform_exclusive_scan(ExPolicy&& policy, FwdIter1 first,
         FwdIter1 last, FwdIter2 dest, T init, BinOp&& binary_op,
         UnOp&& unary_op);
    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/local/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/type_support/unused.hpp>

#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/algorithms/transform_inclusive_scan.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/partitioner.hpp>
#include <pika/parallel/util/scan_partitioner.hpp>
#include <pika/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // transform_exclusive_scan
    namespace detail {
        /// \cond NOINTERNAL

        // Our own version of the sequential transform_exclusive_scan.
        template <typename InIter, typename Sent, typename OutIter,
            typename Conv, typename T, typename Op>
        static constexpr util::in_out_result<InIter, OutIter>
        sequential_transform_exclusive_scan(
            InIter first, Sent last, OutIter dest, Conv&& conv, T init, Op&& op)
        {
            T temp = init;
            for (/* */; first != last; (void) ++first, ++dest)
            {
                init = PIKA_INVOKE(op, init, PIKA_INVOKE(conv, *first));
                *dest = temp;
                temp = init;
            }
            return util::in_out_result<InIter, OutIter>{first, dest};
        }

        template <typename InIter, typename OutIter, typename Conv, typename T,
            typename Op>
        static constexpr T sequential_transform_exclusive_scan_n(InIter first,
            std::size_t count, OutIter dest, Conv&& conv, T init, Op&& op)
        {
            T temp = init;
            for (/* */; count-- != 0; (void) ++first, ++dest)
            {
                init = PIKA_INVOKE(op, init, PIKA_INVOKE(conv, *first));
                *dest = temp;
                temp = init;
            }
            return init;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename IterPair>
        struct transform_exclusive_scan
          : public detail::algorithm<transform_exclusive_scan<IterPair>,
                IterPair>
        {
            transform_exclusive_scan()
              : transform_exclusive_scan::algorithm("transform_exclusive_scan")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename Conv, typename T, typename OutIter, typename Op>
            static constexpr util::in_out_result<InIter, OutIter> sequential(
                ExPolicy, InIter first, Sent last, OutIter dest, Conv&& conv,
                T&& init, Op&& op)
            {
                return sequential_transform_exclusive_scan(first, last, dest,
                    PIKA_FORWARD(Conv, conv), PIKA_FORWARD(T, init),
                    PIKA_FORWARD(Op, op));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2, typename Conv, typename T, typename Op>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first, Sent last,
                FwdIter2 dest, Conv&& conv, T&& init, Op&& op)
            {
                using result_type = util::in_out_result<FwdIter1, FwdIter2>;
                using result =
                    util::detail::algorithm_result<ExPolicy, result_type>;
                using zip_iterator =
                    pika::util::zip_iterator<FwdIter1, FwdIter2>;
                using difference_type =
                    typename std::iterator_traits<FwdIter1>::difference_type;

                if (first == last)
                    return result::get(std::move(result_type{first, dest}));

                FwdIter1 last_iter = first;
                difference_type count =
                    detail::advance_and_get_distance(last_iter, last);

                FwdIter2 final_dest = dest;
                std::advance(final_dest, count);

                // The overall scan algorithm is performed by executing 2
                // subsequent parallel steps. The first calculates the scan
                // results for each partition and the second produces the
                // overall result

                using pika::get;
                using pika::util::make_zip_iterator;

                auto f3 = [op](zip_iterator part_begin, std::size_t part_size,
                              T val) mutable -> void {
                    FwdIter2 dst = get<1>(part_begin.get_iterator_tuple());
                    *dst++ = val;

                    util::loop_n<std::decay_t<ExPolicy>>(
                        dst, part_size - 1, [&op, &val](FwdIter2 it) -> void {
                            *it = PIKA_INVOKE(op, val, *it);
                        });
                };

                return util::scan_partitioner<ExPolicy, result_type, T>::call(
                    PIKA_FORWARD(ExPolicy, policy),
                    make_zip_iterator(first, dest), count, init,
                    // step 1 performs first part of scan algorithm
                    [op, conv](zip_iterator part_begin,
                        std::size_t part_size) mutable -> T {
                        T part_init = PIKA_INVOKE(conv, get<0>(*part_begin++));

                        auto iters = part_begin.get_iterator_tuple();
                        return sequential_transform_exclusive_scan_n(
                            get<0>(iters), part_size - 1, get<1>(iters), conv,
                            part_init, op);
                    },
                    // step 2 propagates the partition results from left
                    // to right
                    op,
                    // step 3 runs final accumulation on each partition
                    PIKA_MOVE(f3),
                    // use this return value
                    [last_iter, final_dest](std::vector<T>&&,
                        std::vector<pika::future<void>>&& data) -> result_type {
                        // make sure iterators embedded in function object that is
                        // attached to futures are invalidated
                        data.clear();

                        return result_type{last_iter, final_dest};
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename Op, typename Conv,
        PIKA_CONCEPT_REQUIRES_(
            pika::is_execution_policy<ExPolicy>::value &&
            pika::traits::is_iterator_v<FwdIter1> &&
            pika::traits::is_iterator_v<FwdIter2> &&
            pika::is_invocable_v<Conv,
                    typename std::iterator_traits<FwdIter1>::value_type> &&
            pika::is_invocable_v<Op,
                    typename pika::util::invoke_result_t<Conv,
                        typename std::iterator_traits<FwdIter1>::value_type>,
                    typename pika::util::invoke_result_t<Conv,
                        typename std::iterator_traits<FwdIter1>::value_type>
            >
        )>
    // clang-format on
    PIKA_DEPRECATED_V(0, 1,
        "pika::parallel::transform_exclusive_scan is deprecated, use "
        "pika::transform_exclusive_scan instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        transform_exclusive_scan(ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest, T init, Op&& op, Conv&& conv)
    {
        static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
            "Requires at least forward iterator.");
        static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
            "Requires at least forward iterator.");

        using result_type = parallel::util::in_out_result<FwdIter1, FwdIter2>;

#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return parallel::util::get_second_element(
            detail::transform_exclusive_scan<result_type>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last, dest,
                PIKA_FORWARD(Conv, conv), PIKA_MOVE(init), PIKA_FORWARD(Op, op)));
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace pika::parallel::v1

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for pika::transform_exclusive_scan
    inline constexpr struct transform_exclusive_scan_t final
      : pika::detail::tag_parallel_algorithm<transform_exclusive_scan_t>
    {
        // clang-format off
        template <typename InIter, typename OutIter,
            typename BinOp, typename UnOp,
            typename T = typename std::iterator_traits<InIter>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<InIter> &&
                pika::traits::is_iterator_v<OutIter> &&
                pika::is_invocable_v<UnOp,
                    typename std::iterator_traits<InIter>::value_type> &&
                pika::is_invocable_v<BinOp,
                    typename pika::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<InIter>::value_type>,
                    typename pika::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<InIter>::value_type>
                >
            )>
        // clang-format on
        friend OutIter tag_fallback_invoke(pika::transform_exclusive_scan_t,
            InIter first, InIter last, OutIter dest, T init, BinOp&& binary_op,
            UnOp&& unary_op)
        {
            static_assert((pika::traits::is_input_iterator_v<InIter>),
                "Requires at least input iterator.");
            static_assert((pika::traits::is_output_iterator_v<OutIter>),
                "Requires at least output iterator.");

            using result_type = parallel::util::in_out_result<InIter, OutIter>;

            return parallel::util::get_second_element(
                pika::parallel::v1::detail::transform_exclusive_scan<
                    result_type>()
                    .call(pika::execution::seq, first, last, dest,
                        PIKA_FORWARD(UnOp, unary_op), PIKA_MOVE(init),
                        PIKA_FORWARD(BinOp, binary_op)));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename BinOp, typename UnOp,
            typename T = typename std::iterator_traits<FwdIter1>::value_type,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2> &&
                pika::is_invocable_v<UnOp,
                    typename std::iterator_traits<FwdIter1>::value_type> &&
                pika::is_invocable_v<BinOp,
                    typename pika::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>,
                    typename pika::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_invoke(pika::transform_exclusive_scan_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, T init,
            BinOp&& binary_op, UnOp&& unary_op)
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(pika::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            using result_type =
                parallel::util::in_out_result<FwdIter1, FwdIter2>;

            return parallel::util::get_second_element(
                pika::parallel::v1::detail::transform_exclusive_scan<
                    result_type>()
                    .call(PIKA_FORWARD(ExPolicy, policy), first, last, dest,
                        PIKA_FORWARD(UnOp, unary_op), PIKA_MOVE(init),
                        PIKA_FORWARD(BinOp, binary_op)));
        }
    } transform_exclusive_scan{};
}    // namespace pika

#endif    // DOXYGEN
