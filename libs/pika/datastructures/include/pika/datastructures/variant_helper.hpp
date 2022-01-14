//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/datastructures/config/defines.hpp>

#include <pika/type_support/pack.hpp>

#include <cstddef>    // for size_t
#include <utility>
#include <variant>

namespace pika {

    // This is put into the same embedded namespace as the implementations in
    // tuple.hpp
    namespace adl_barrier {

        template <std::size_t I, typename... Ts>
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE
            typename util::at_index<I, Ts...>::type&
            get(std::variant<Ts...>& var) noexcept
        {
            return std::get<I>(var);
        }

        template <std::size_t I, typename... Ts>
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE
            typename util::at_index<I, Ts...>::type const&
            get(std::variant<Ts...> const& var) noexcept
        {
            return std::get<I>(var);
        }

        template <std::size_t I, typename... Ts>
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE
            typename util::at_index<I, Ts...>::type&&
            get(std::variant<Ts...>&& var) noexcept
        {
            return std::get<I>(PIKA_MOVE(var));
        }

        template <std::size_t I, typename... Ts>
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE
            typename util::at_index<I, Ts...>::type const&&
            get(std::variant<Ts...> const&& var) noexcept
        {
            return std::get<I>(PIKA_MOVE(var));
        }
    }    // namespace adl_barrier

    using pika::adl_barrier::get;
}    // namespace pika
