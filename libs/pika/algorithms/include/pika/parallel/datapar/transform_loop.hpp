//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)
#include <pika/datastructures/tuple.hpp>
#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/executors/datapar/execution_policy.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/parallel/datapar/iterator_helpers.hpp>
#include <pika/parallel/util/cancellation_token.hpp>
#include <pika/parallel/util/transform_loop.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace parallel { namespace util {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_transform_loop_n
        {
            typedef typename std::decay<Iterator>::type iterator_type;

            typedef typename traits::vector_pack_type<
                typename std::iterator_traits<iterator_type>::value_type>::type
                V;

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest, F&& f)
            {
                std::size_t len = count;

                for (/* */;
                     !(is_data_aligned(first) && is_data_aligned(dest)) &&
                     len != 0;
                     --len)
                {
                    datapar_transform_loop_step::call1(f, first, dest);
                }

                static constexpr std::size_t size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    datapar_transform_loop_step::callv(f, first, dest);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_transform_loop_step::call1(f, first, dest);
                }

                return std::make_pair(PIKA_MOVE(first), PIKA_MOVE(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest, F&& f)
            {
                return util::transform_loop_n<pika::execution::sequenced_policy>(
                    first, count, dest, PIKA_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        std::pair<Iter, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_loop_n_t<ExPolicy>, Iter it,
        std::size_t count, OutIter dest, F&& f)
    {
        return detail::datapar_transform_loop_n<Iter>::call(
            it, count, dest, PIKA_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_transform_loop_n_ind
        {
            typedef typename std::decay<Iterator>::type iterator_type;

            typedef typename traits::vector_pack_type<
                typename std::iterator_traits<iterator_type>::value_type>::type
                V;

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest, F&& f)
            {
                std::size_t len = count;

                for (/* */;
                     !(is_data_aligned(first) && is_data_aligned(dest)) &&
                     len != 0;
                     --len)
                {
                    datapar_transform_loop_step_ind::call1(f, first, dest);
                }

                static constexpr std::size_t size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    datapar_transform_loop_step_ind::callv(f, first, dest);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_transform_loop_step_ind::call1(f, first, dest);
                }

                return std::make_pair(PIKA_MOVE(first), PIKA_MOVE(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, std::size_t count, OutIter dest, F&& f)
            {
                return util::transform_loop_n_ind<
                    pika::execution::sequenced_policy>(
                    first, count, dest, PIKA_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename Iter, typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        std::pair<Iter, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_loop_n_ind_t<ExPolicy>, Iter it,
        std::size_t count, OutIter dest, F&& f)
    {
        return detail::datapar_transform_loop_n_ind<Iter>::call(
            it, count, dest, PIKA_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_transform_loop
        {
            typedef typename std::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;
            typedef typename traits::vector_pack_type<value_type, 1>::type V1;

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, InIter last, OutIter dest, F&& f)
            {
                return util::transform_loop_n<pika::execution::simd_policy>(
                    first, std::distance(first, last), dest, PIKA_FORWARD(F, f));
            }

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, InIter last, OutIter dest, F&& f)
            {
                return util::transform_loop(
                    pika::execution::seq, first, last, dest, PIKA_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename IterB, typename IterE, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(pika::parallel::util::transform_loop_t,
            pika::execution::simd_policy, IterB it, IterE end, OutIter dest,
            F&& f)
    {
        auto ret = detail::datapar_transform_loop<IterB>::call(
            it, end, dest, PIKA_FORWARD(F, f));

        return util::in_out_result<IterB, OutIter>{
            PIKA_MOVE(ret.first), PIKA_MOVE(ret.second)};
    }

    template <typename IterB, typename IterE, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(pika::parallel::util::transform_loop_t,
            pika::execution::simd_task_policy, IterB it, IterE end, OutIter dest,
            F&& f)
    {
        auto ret = detail::datapar_transform_loop<IterB>::call(
            it, end, dest, PIKA_FORWARD(F, f));

        return util::in_out_result<IterB, OutIter>{
            PIKA_MOVE(ret.first), PIKA_MOVE(ret.second)};
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_transform_loop_ind
        {
            typedef typename std::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;
            typedef typename traits::vector_pack_type<value_type, 1>::type V1;

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter, OutIter>::value &&
                    iterator_datapar_compatible<InIter>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, InIter last, OutIter dest, F&& f)
            {
                return util::transform_loop_n_ind<pika::execution::simd_policy>(
                    first, std::distance(first, last), dest, PIKA_FORWARD(F, f));
            }

            template <typename InIter, typename OutIter, typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter, OutIter>::value ||
                    !iterator_datapar_compatible<InIter>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                std::pair<InIter, OutIter>>::type
            call(InIter first, InIter last, OutIter dest, F&& f)
            {
                auto ret = util::transform_loop_ind(
                    pika::execution::seq, first, last, dest, PIKA_FORWARD(F, f));
                return std::pair<InIter, OutIter>{
                    PIKA_MOVE(ret.in), PIKA_MOVE(ret.out)};
            }
        };
    }    // namespace detail

    template <typename IterB, typename IterE, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(pika::parallel::util::transform_loop_ind_t,
            pika::execution::simd_policy, IterB it, IterE end, OutIter dest,
            F&& f)
    {
        auto ret = detail::datapar_transform_loop_ind<IterB>::call(
            it, end, dest, PIKA_FORWARD(F, f));

        return util::in_out_result<IterB, OutIter>{
            PIKA_MOVE(ret.first), PIKA_MOVE(ret.second)};
    }

    template <typename IterB, typename IterE, typename OutIter, typename F>
    PIKA_HOST_DEVICE
        PIKA_FORCEINLINE constexpr util::in_out_result<IterB, OutIter>
        tag_invoke(pika::parallel::util::transform_loop_ind_t,
            pika::execution::simd_task_policy, IterB it, IterE end, OutIter dest,
            F&& f)
    {
        auto ret = detail::datapar_transform_loop_ind<IterB>::call(
            it, end, dest, PIKA_FORWARD(F, f));

        return util::in_out_result<IterB, OutIter>{
            PIKA_MOVE(ret.first), PIKA_MOVE(ret.second)};
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop_n
        {
            typedef typename std::decay<Iter1>::type iterator1_type;
            typedef typename std::iterator_traits<iterator1_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                pika::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                std::size_t len = count;

                for (/* */;
                     !(is_data_aligned(first1) && is_data_aligned(first2) &&
                         is_data_aligned(dest)) &&
                     len != 0;
                     --len)
                {
                    datapar_transform_loop_step::call1(f, first1, first2, dest);
                }

                static constexpr std::size_t size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    datapar_transform_loop_step::callv(f, first1, first2, dest);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_transform_loop_step::call1(f, first1, first2, dest);
                }

                return pika::make_tuple(
                    PIKA_MOVE(first1), PIKA_MOVE(first2), PIKA_MOVE(dest));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                pika::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                return util::transform_binary_loop_n<
                    pika::execution::sequenced_policy>(
                    first1, count, first2, dest, PIKA_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        pika::tuple<InIter1, InIter2, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_binary_loop_n_t<ExPolicy>,
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop_n<InIter1, InIter2>::call(
            first1, count, first2, dest, PIKA_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop
        {
            typedef typename std::decay<Iter1>::type iterator1_type;
            typedef typename std::decay<Iter2>::type iterator2_type;

            typedef typename std::iterator_traits<iterator1_type>::value_type
                value1_type;
            typedef typename std::iterator_traits<iterator2_type>::value_type
                value2_type;

            typedef typename traits::vector_pack_type<value1_type, 1>::type V11;
            typedef typename traits::vector_pack_type<value2_type, 1>::type V12;

            typedef typename traits::vector_pack_type<value1_type>::type V1;
            typedef typename traits::vector_pack_type<value2_type>::type V2;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F&& f)
            {
                auto ret = util::transform_binary_loop_n<
                    pika::execution::par_simd_policy>(first1,
                    std::distance(first1, last1), first2, dest,
                    PIKA_FORWARD(F, f));

                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    pika::get<0>(ret), pika::get<1>(ret), pika::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F&& f)
            {
                return util::transform_binary_loop<
                    pika::execution::sequenced_policy>(
                    first1, last1, first2, dest, PIKA_FORWARD(F, f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F&& f)
            {
                // different versions of clang-format do different things
                // clang-format off
                std::size_t count = (std::min) (std::distance(first1, last1),
                    std::distance(first2, last2));
                // clang-format on

                auto ret = util::transform_binary_loop_n<
                    pika::execution::par_simd_policy>(
                    first1, count, first2, dest, PIKA_FORWARD(F, f));

                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    pika::get<0>(ret), pika::get<1>(ret), pika::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F&& f)
            {
                return util::transform_binary_loop<
                    pika::execution::sequenced_policy>(
                    first1, last1, first2, last2, dest, PIKA_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_binary_loop_t<ExPolicy>,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, dest, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_binary_loop_t<ExPolicy>,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, last2, dest, PIKA_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop_ind_n
        {
            typedef typename std::decay<Iter1>::type iterator1_type;
            typedef typename std::iterator_traits<iterator1_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                pika::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                std::size_t len = count;

                for (/* */;
                     !(is_data_aligned(first1) && is_data_aligned(first2) &&
                         is_data_aligned(dest)) &&
                     len != 0;
                     --len)
                {
                    datapar_transform_loop_step_ind::call1(
                        f, first1, first2, dest);
                }

                static constexpr std::size_t size =
                    traits::vector_pack_size<V>::value;

                for (std::int64_t len_v = std::int64_t(len - (size + 1));
                     len_v > 0; len_v -= size, len -= size)
                {
                    datapar_transform_loop_step_ind::callv(
                        f, first1, first2, dest);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_transform_loop_step_ind::call1(
                        f, first1, first2, dest);
                }

                return pika::make_tuple(
                    PIKA_MOVE(first1), PIKA_MOVE(first2), PIKA_MOVE(dest));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                pika::tuple<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F&& f)
            {
                return util::transform_binary_loop_ind_n<
                    pika::execution::sequenced_policy>(
                    first1, count, first2, dest, PIKA_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        pika::tuple<InIter1, InIter2, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_binary_loop_ind_n_t<ExPolicy>,
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop_ind_n<InIter1,
            InIter2>::call(first1, count, first2, dest, PIKA_FORWARD(F, f));
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop_ind
        {
            typedef typename std::decay<Iter1>::type iterator1_type;
            typedef typename std::decay<Iter2>::type iterator2_type;

            typedef typename std::iterator_traits<iterator1_type>::value_type
                value1_type;
            typedef typename std::iterator_traits<iterator2_type>::value_type
                value2_type;

            typedef typename traits::vector_pack_type<value1_type, 1>::type V11;
            typedef typename traits::vector_pack_type<value2_type, 1>::type V12;

            typedef typename traits::vector_pack_type<value1_type>::type V1;
            typedef typename traits::vector_pack_type<value2_type>::type V2;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F&& f)
            {
                auto ret = util::transform_binary_loop_ind_n<
                    pika::execution::par_simd_policy>(first1,
                    std::distance(first1, last1), first2, dest,
                    PIKA_FORWARD(F, f));

                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    pika::get<0>(ret), pika::get<1>(ret), pika::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F&& f)
            {
                return util::transform_binary_loop_ind<
                    pika::execution::sequenced_policy>(
                    first1, last1, first2, dest, PIKA_FORWARD(F, f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value &&
                    iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F&& f)
            {
                std::size_t count = (std::min)(
                    std::distance(first1, last1), std::distance(first2, last2));

                auto ret = util::transform_binary_loop_ind_n<
                    pika::execution::par_simd_policy>(
                    first1, count, first2, dest, PIKA_FORWARD(F, f));

                return util::in_in_out_result<InIter1, InIter2, OutIter>{
                    pika::get<0>(ret), pika::get<1>(ret), pika::get<2>(ret)};
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, OutIter>::value ||
                    !iterators_datapar_compatible<InIter2, OutIter>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value ||
                    !iterator_datapar_compatible<OutIter>::value,
                util::in_in_out_result<InIter1, InIter2, OutIter>>::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F&& f)
            {
                return util::transform_binary_loop_ind<
                    pika::execution::sequenced_policy>(
                    first1, last1, first2, last2, dest, PIKA_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_binary_loop_ind_t<ExPolicy>,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop_ind<InIter1,
            InIter2>::call(first1, last1, first2, dest, PIKA_FORWARD(F, f));
    }

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE typename std::enable_if<
        pika::is_vectorpack_execution_policy<ExPolicy>::value,
        util::in_in_out_result<InIter1, InIter2, OutIter>>::type
    tag_invoke(pika::parallel::util::transform_binary_loop_ind_t<ExPolicy>,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F&& f)
    {
        return detail::datapar_transform_binary_loop_ind<InIter1,
            InIter2>::call(first1, last1, first2, last2, dest,
            PIKA_FORWARD(F, f));
    }
}}}    // namespace pika::parallel::util
#endif
