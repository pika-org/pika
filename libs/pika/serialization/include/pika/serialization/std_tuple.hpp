//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2020 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//  Copyright (c) 2019 Mikael Simberg
//  Copyright (c) 2020-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/serialization/serialization_fwd.hpp>
#include <pika/serialization/traits/is_bitwise_serializable.hpp>
#include <pika/serialization/traits/is_not_bitwise_serializable.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace pika { namespace traits {

    template <typename... Ts>
    struct is_bitwise_serializable<std::tuple<Ts...>>
      : ::pika::util::all_of<pika::traits::is_bitwise_serializable<
            typename std::remove_const<Ts>::type>...>
    {
    };

    template <typename... Ts>
    struct is_not_bitwise_serializable<std::tuple<Ts...>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<std::tuple<Ts...>>>
    {
    };
}}    // namespace pika::traits

namespace pika { namespace serialization {

    namespace detail {

        template <typename Archive, typename Is, typename... Ts>
        struct std_serialize_with_index_pack;

        template <typename Archive, std::size_t... Is, typename... Ts>
        struct std_serialize_with_index_pack<Archive,
            pika::util::index_pack<Is...>, Ts...>
        {
            static void call(Archive& ar, std::tuple<Ts...>& t, unsigned int)
            {
#if !defined(PIKA_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS)
                int const _sequencer[] = {((ar & std::get<Is>(t)), 0)...};
#else
                int const _sequencer[] = {
                    ((ar &
                         const_cast<std::remove_const_t<Ts>&>(std::get<Is>(t))),
                        0)...};
#endif
                (void) _sequencer;
            }
        };
    }    // namespace detail

    template <typename Archive, typename... Ts>
    void serialize(Archive& ar, std::tuple<Ts...>& t, unsigned int version)
    {
        using Is = typename pika::util::make_index_pack<sizeof...(Ts)>::type;
        detail::std_serialize_with_index_pack<Archive, Is, Ts...>::call(
            ar, t, version);
    }

    template <typename Archive>
    void serialize(Archive&, std::tuple<>&, unsigned int)
    {
    }

}}    // namespace pika::serialization
