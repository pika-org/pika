//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

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
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename... Iter>
    struct is_data_aligned_impl<pika::util::zip_iterator<Iter...>>
    {
        template <std::size_t... Is>
        static PIKA_FORCEINLINE bool call(
            pika::util::zip_iterator<Iter...> const& it,
            pika::util::index_pack<Is...>)
        {
            auto const& t = it.get_iterator_tuple();
            bool const sequencer[] = {
                true, is_data_aligned(pika::get<Is>(t))...};
            return std::all_of(&sequencer[1], &sequencer[sizeof...(Is) + 1],
                [](bool val) { return val; });
        }

        static PIKA_FORCEINLINE bool call(
            pika::util::zip_iterator<Iter...> const& it)
        {
            return call(it,
                typename pika::util::make_index_pack<sizeof...(Iter)>::type());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Iter>
    struct iterator_datapar_compatible_impl<pika::util::zip_iterator<Iter...>>
      : pika::util::all_of<std::is_arithmetic<
            typename std::iterator_traits<Iter>::value_type>...>
    {
    };
}}}}    // namespace pika::parallel::util::detail

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Tuple, typename... Iter, std::size_t... Is>
        Tuple aligned_pack(pika::util::zip_iterator<Iter...> const& iter,
            pika::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return pika::make_tuple(
                vector_pack_load<typename pika::tuple_element<Is, Tuple>::type,
                    typename std::iterator_traits<Iter>::value_type>::
                    aligned(pika::get<Is>(t))...);
        }

        template <typename Tuple, typename... Iter, std::size_t... Is>
        Tuple unaligned_pack(pika::util::zip_iterator<Iter...> const& iter,
            pika::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return pika::make_tuple(
                vector_pack_load<typename pika::tuple_element<Is, Tuple>::type,
                    typename std::iterator_traits<Iter>::value_type>::
                    unaligned(pika::get<Is>(t))...);
        }
    }    // namespace detail

    template <typename... Vector, typename ValueType>
    struct vector_pack_load<pika::tuple<Vector...>, ValueType>
    {
        typedef pika::tuple<Vector...> value_type;

        template <typename... Iter>
        static value_type aligned(pika::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::aligned_pack<value_type>(iter,
                typename pika::util::make_index_pack<sizeof...(Iter)>::type());
        }

        template <typename... Iter>
        static value_type unaligned(
            pika::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::unaligned_pack<value_type>(iter,
                typename pika::util::make_index_pack<sizeof...(Iter)>::type());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Tuple, typename... Iter, std::size_t... Is>
        void aligned_pack(Tuple& value,
            pika::util::zip_iterator<Iter...> const& iter,
            pika::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            int const sequencer[] = {0,
                (vector_pack_store<typename pika::tuple_element<Is, Tuple>::type,
                     typename std::iterator_traits<Iter>::value_type>::
                        aligned(pika::get<Is>(value), pika::get<Is>(t)),
                    0)...};
            (void) sequencer;
        }

        template <typename Tuple, typename... Iter, std::size_t... Is>
        void unaligned_pack(Tuple& value,
            pika::util::zip_iterator<Iter...> const& iter,
            pika::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            int const sequencer[] = {0,
                (vector_pack_store<typename pika::tuple_element<Is, Tuple>::type,
                     typename std::iterator_traits<Iter>::value_type>::
                        unaligned(pika::get<Is>(value), pika::get<Is>(t)),
                    0)...};
            (void) sequencer;
        }
    }    // namespace detail

    template <typename... Vector, typename ValueType>
    struct vector_pack_store<pika::tuple<Vector...>, ValueType>
    {
        template <typename V, typename... Iter>
        static void aligned(
            V& value, pika::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::aligned_pack(value, iter,
                typename pika::util::make_index_pack<sizeof...(Iter)>::type());
        }

        template <typename V, typename... Iter>
        static void unaligned(
            V& value, pika::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::unaligned_pack(value, iter,
                typename pika::util::make_index_pack<sizeof...(Iter)>::type());
        }
    };
}}}    // namespace pika::parallel::traits

#endif
