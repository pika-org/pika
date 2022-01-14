//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/datastructures/member_pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika {
    namespace execution {
        namespace experimental {
            namespace detail {
    template <typename Tag, typename IsPack, typename... Ts>
    struct partial_algorithm_base;

    template <typename Tag, std::size_t... Is, typename... Ts>
    struct partial_algorithm_base<Tag, pika::util::index_pack<Is...>, Ts...>
    {
    private:
        pika::util::member_pack_for<Ts...> ts;

    public:
        template <typename... Ts_>
        explicit constexpr partial_algorithm_base(Ts_&&... ts)
          : ts(std::piecewise_construct, PIKA_FORWARD(Ts_, ts)...)
        {
        }

        partial_algorithm_base(partial_algorithm_base&&) = default;
        partial_algorithm_base& operator=(partial_algorithm_base&&) = default;
        partial_algorithm_base(partial_algorithm_base const&) = delete;
        partial_algorithm_base& operator=(
            partial_algorithm_base const&) = delete;

        template <typename U>
        friend constexpr PIKA_FORCEINLINE auto operator|(
            U&& u, partial_algorithm_base p)
        {
            return Tag{}(
                PIKA_FORWARD(U, u), PIKA_MOVE(p.ts).template get<Is>()...);
        }
    };

    template <typename Tag, typename... Ts>
    using partial_algorithm = partial_algorithm_base<Tag,
        typename pika::util::make_index_pack<sizeof...(Ts)>::type, Ts...>;
}}}}    // namespace pika::execution::experimental::detail
