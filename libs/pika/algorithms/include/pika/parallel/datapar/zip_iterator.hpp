//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)

#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/iterator_support/zip_iterator.hpp>
#include <pika/type_support/pack.hpp>

#include <pika/execution/traits/vector_pack_alignment_size.hpp>
#include <pika/execution/traits/vector_pack_load_store.hpp>
#include <pika/execution/traits/vector_pack_type.hpp>
#include <pika/parallel/datapar/iterator_helpers.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>

namespace pika::parallel::detail {
    template <typename... Iter>
    struct is_data_aligned_impl<pika::util::zip_iterator<Iter...>>
    {
        template <std::size_t... Is>
        static PIKA_FORCEINLINE bool call(
            pika::util::zip_iterator<Iter...> const& it,
            pika::util::detail::index_pack<Is...>)
        {
            auto const& t = it.get_iterator_tuple();
            bool const sequencer[] = {
                true, is_data_aligned(std::get<Is>(t))...};
            return std::all_of(&sequencer[1], &sequencer[sizeof...(Is) + 1],
                [](bool val) { return val; });
        }

        static PIKA_FORCEINLINE bool call(
            pika::util::zip_iterator<Iter...> const& it)
        {
            return call(
                it, pika::util::detail::make_index_pack_t<sizeof...(Iter)>());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Iter>
    struct iterator_datapar_compatible_impl<pika::util::zip_iterator<Iter...>>
      : pika::util::detail::all_of<std::is_arithmetic<
            typename std::iterator_traits<Iter>::value_type>...>
    {
    };
}    // namespace pika::parallel::detail

namespace pika::parallel::traits::detail {
    template <typename Tuple, typename... Iter, std::size_t... Is>
    Tuple aligned_pack(pika::util::zip_iterator<Iter...> const& iter,
        pika::util::detail::index_pack<Is...>)
    {
        auto const& t = iter.get_iterator_tuple();
        return std::make_tuple(
            vector_pack_load<typename std::tuple_element<Is, Tuple>::type,
                typename std::iterator_traits<Iter>::value_type>::
                aligned(std::get<Is>(t))...);
    }

    template <typename Tuple, typename... Iter, std::size_t... Is>
    Tuple unaligned_pack(pika::util::zip_iterator<Iter...> const& iter,
        pika::util::detail::index_pack<Is...>)
    {
        auto const& t = iter.get_iterator_tuple();
        return std::make_tuple(
            vector_pack_load<typename std::tuple_element<Is, Tuple>::type,
                typename std::iterator_traits<Iter>::value_type>::
                unaligned(std::get<Is>(t))...);
    }

    template <typename... Vector, typename ValueType>
    struct vector_pack_load<std::tuple<Vector...>, ValueType>
    {
        using value_type = std::tuple<Vector...>;

        template <typename... Iter>
        static value_type aligned(pika::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::aligned_pack<value_type>(
                iter, pika::util::detail::make_index_pack_t<sizeof...(Iter)>());
        }

        template <typename... Iter>
        static value_type unaligned(
            pika::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::unaligned_pack<value_type>(
                iter, pika::util::detail::make_index_pack<sizeof...(Iter)>());
        }
    };

    template <typename Tuple, typename... Iter, std::size_t... Is>
    void aligned_pack(Tuple& value,
        pika::util::zip_iterator<Iter...> const& iter,
        pika::util::detail::index_pack<Is...>)
    {
        auto const& t = iter.get_iterator_tuple();
        int const sequencer[] = {0,
            (vector_pack_store<typename std::tuple_element<Is, Tuple>::type,
                 typename std::iterator_traits<Iter>::value_type>::
                    aligned(std::get<Is>(value), std::get<Is>(t)),
                0)...};
        (void) sequencer;
    }

    template <typename Tuple, typename... Iter, std::size_t... Is>
    void unaligned_pack(Tuple& value,
        pika::util::zip_iterator<Iter...> const& iter,
        pika::util::detail::index_pack<Is...>)
    {
        auto const& t = iter.get_iterator_tuple();
        int const sequencer[] = {0,
            (vector_pack_store<typename std::tuple_element<Is, Tuple>::type,
                 typename std::iterator_traits<Iter>::value_type>::
                    unaligned(std::get<Is>(value), std::get<Is>(t)),
                0)...};
        (void) sequencer;
    }

    template <typename... Vector, typename ValueType>
    struct vector_pack_store<std::tuple<Vector...>, ValueType>
    {
        template <typename V, typename... Iter>
        static void aligned(
            V& value, pika::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::aligned_pack(value, iter,
                pika::util::detail::make_index_pack_t<sizeof...(Iter)>());
        }

        template <typename V, typename... Iter>
        static void unaligned(
            V& value, pika::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::unaligned_pack(value, iter,
                pika::util::detail::make_index_pack<sizeof...(Iter)>());
        }
    };
}    // namespace pika::parallel::traits::detail

#endif
