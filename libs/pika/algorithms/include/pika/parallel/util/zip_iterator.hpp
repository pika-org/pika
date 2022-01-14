//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/futures/future.hpp>
#include <pika/iterator_support/zip_iterator.hpp>
#include <pika/modules/execution.hpp>
#include <pika/parallel/util/result_types.hpp>

#include <utility>

namespace pika { namespace parallel { inline namespace v1 { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename R, typename ZipIter>
    R get_iter(ZipIter&& zipiter)
    {
        return pika::get<N>(zipiter.get_iterator_tuple());
    }

    template <int N, typename R, typename ZipIter>
    R get_iter(pika::future<ZipIter>&& zipiter)
    {
        typedef typename pika::tuple_element<N,
            typename ZipIter::iterator_tuple_type>::type result_type;

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
    pika::future<typename ZipIter::iterator_tuple_type> get_iter_tuple(
        pika::future<ZipIter>&& zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type result_type;
        return pika::make_future<result_type>(PIKA_MOVE(zipiter),
            [](ZipIter zipiter) { return get_iter_tuple(PIKA_MOVE(zipiter)); });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    std::pair<typename pika::tuple_element<0,
                  typename ZipIter::iterator_tuple_type>::type,
        typename pika::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>
    get_iter_pair(ZipIter&& zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type iterator_tuple_type;

        iterator_tuple_type t = zipiter.get_iterator_tuple();
        return std::make_pair(pika::get<0>(t), pika::get<1>(t));
    }

    template <typename ZipIter>
    pika::future<std::pair<typename pika::tuple_element<0,
                              typename ZipIter::iterator_tuple_type>::type,
        typename pika::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>>
    get_iter_pair(pika::future<ZipIter>&& zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type iterator_tuple_type;

        typedef std::pair<
            typename pika::tuple_element<0, iterator_tuple_type>::type,
            typename pika::tuple_element<1, iterator_tuple_type>::type>
            result_type;

        return pika::make_future<result_type>(PIKA_MOVE(zipiter),
            [](ZipIter zipiter) { return get_iter_pair(PIKA_MOVE(zipiter)); });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    util::in_in_result<typename pika::tuple_element<0,
                           typename ZipIter::iterator_tuple_type>::type,
        typename pika::tuple_element<1,
            typename ZipIter::iterator_tuple_type>::type>
    get_iter_in_in_result(ZipIter&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type = util::in_in_result<
            typename pika::tuple_element<0, iterator_tuple_type>::type,
            typename pika::tuple_element<1, iterator_tuple_type>::type>;

        iterator_tuple_type t = zipiter.get_iterator_tuple();
        return result_type{pika::get<0>(t), pika::get<1>(t)};
    }

    template <typename ZipIter>
    pika::future<
        util::in_in_result<typename pika::tuple_element<0,
                               typename ZipIter::iterator_tuple_type>::type,
            typename pika::tuple_element<1,
                typename ZipIter::iterator_tuple_type>::type>>
    get_iter_in_in_result(pika::future<ZipIter>&& zipiter)
    {
        using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

        using result_type = util::in_in_result<
            typename pika::tuple_element<0, iterator_tuple_type>::type,
            typename pika::tuple_element<1, iterator_tuple_type>::type>;

        return pika::make_future<result_type>(
            PIKA_MOVE(zipiter), [](ZipIter zipiter) {
                return get_iter_in_in_result(PIKA_MOVE(zipiter));
            });
    }
}}}}    // namespace pika::parallel::v1::detail
