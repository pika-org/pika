//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/futures/future.hpp>
#include <pika/iterator_support/zip_iterator.hpp>
#include <pika/modules/execution.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <tuple>
#include <utility>

namespace pika::parallel::detail {
    template <int N, typename R, typename ZipIter>
    R get_iter(ZipIter&& zipiter)
    {
        return std::get<N>(zipiter.get_iterator_tuple());
    }

    template <int N, typename R, typename ZipIter>
    R get_iter(pika::future<ZipIter>&& zipiter)
    {
        using result_type = typename std::tuple_element<N,
            typename ZipIter::iterator_tuple_type>::type;

        return pika::make_future<result_type>(
            PIKA_MOVE(zipiter), [](ZipIter zipiter) {
                return get_iter<N, result_type>(PIKA_MOVE(zipiter));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    typename ZipIter::iterator_tuple_type get_iter_tuple(ZipIter&& zipiter)
    {
        return zipiter.get_iterator_tuple();
    }

    template <typename ZipIter>
    pika::future<typename ZipIter::iterator_tuple_type>
    get_iter_tuple(pika::future<ZipIter>&& zipiter)
    {
        using result_type = typename ZipIter::iterator_tuple_type;
        return pika::make_future<result_type>(PIKA_MOVE(zipiter),
            [](ZipIter zipiter) { return get_iter_tuple(PIKA_MOVE(zipiter)); });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    std::pair<typename std::tuple_element<0,
                  typename ZipIter::iterator_tuple_type>::type,
        typename std::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>
    get_iter_pair(ZipIter&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        iterator_tuple_type t = zipiter.get_iterator_tuple();
        return std::make_pair(std::get<0>(t), std::get<1>(t));
    }

    template <typename ZipIter>
    pika::future<std::pair<typename std::tuple_element<0,
                               typename ZipIter::iterator_tuple_type>::type,
        typename std::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>>
    get_iter_pair(pika::future<ZipIter>&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type =
            std::pair<typename std::tuple_element<0, iterator_tuple_type>::type,
                typename std::tuple_element<1, iterator_tuple_type>::type>;

        return pika::make_future<result_type>(PIKA_MOVE(zipiter),
            [](ZipIter zipiter) { return get_iter_pair(PIKA_MOVE(zipiter)); });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    in_in_result<typename std::tuple_element<0,
                     typename ZipIter::iterator_tuple_type>::type,
        typename std::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>
    get_iter_in_in_result(ZipIter&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type = in_in_result<
            typename std::tuple_element<0, iterator_tuple_type>::type,
            typename std::tuple_element<1, iterator_tuple_type>::type>;

        iterator_tuple_type t = zipiter.get_iterator_tuple();
        return result_type{std::get<0>(t), std::get<1>(t)};
    }

    template <typename ZipIter>
    pika::future<in_in_result<typename std::tuple_element<0,
                                  typename ZipIter::iterator_tuple_type>::type,
        typename std::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>>
    get_iter_in_in_result(pika::future<ZipIter>&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type = in_in_result<
            typename std::tuple_element<0, iterator_tuple_type>::type,
            typename std::tuple_element<1, iterator_tuple_type>::type>;

        return pika::make_future<result_type>(
            PIKA_MOVE(zipiter), [](ZipIter zipiter) {
                return get_iter_in_in_result(PIKA_MOVE(zipiter));
            });
    }
}    // namespace pika::parallel::detail
