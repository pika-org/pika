//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
#include <pika/execution_base/p2300_forward.hpp>
#else
#include <pika/datastructures/member_pack.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    namespace just_detail {
        template <typename Is, typename... Ts>
        struct just_sender_impl;

        template <typename std::size_t... Is, typename... Ts>
        struct just_sender_impl<pika::util::index_pack<Is...>, Ts...>
        {
            struct just_sender_type
            {
                pika::util::detail::member_pack_for<std::decay_t<Ts>...> ts;

                constexpr just_sender_type() = default;

                template <typename T,
                    typename = std::enable_if_t<
                        !std::is_same_v<std::decay_t<T>, just_sender_type>>>
                explicit constexpr just_sender_type(T&& t)
                  : ts(std::piecewise_construct, PIKA_FORWARD(T, t))
                {
                }

                template <typename T0, typename T1, typename... Ts_>
                explicit constexpr just_sender_type(
                    T0&& t0, T1&& t1, Ts_&&... ts)
                  : ts(std::piecewise_construct, PIKA_FORWARD(T0, t0),
                        PIKA_FORWARD(T1, t1), PIKA_FORWARD(Ts_, ts)...)
                {
                }

                just_sender_type(just_sender_type&&) = default;
                just_sender_type(just_sender_type const&) = default;
                just_sender_type& operator=(just_sender_type&&) = default;
                just_sender_type& operator=(just_sender_type const&) = default;

                template <template <typename...> class Tuple,
                    template <typename...> class Variant>
                using value_types = Variant<Tuple<std::decay_t<Ts>...>>;

                template <template <typename...> class Variant>
                using error_types = Variant<std::exception_ptr>;

                static constexpr bool sends_done = false;

                template <typename Receiver>
                struct operation_state
                {
                    PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                    pika::util::detail::member_pack_for<std::decay_t<Ts>...> ts;

                    template <typename Receiver_>
                    operation_state(Receiver_&& receiver,
                        pika::util::detail::member_pack_for<std::decay_t<Ts>...>
                            ts)
                      : receiver(PIKA_FORWARD(Receiver_, receiver))
                      , ts(PIKA_MOVE(ts))
                    {
                    }

                    operation_state(operation_state&&) = delete;
                    operation_state& operator=(operation_state&&) = delete;
                    operation_state(operation_state const&) = delete;
                    operation_state& operator=(operation_state const&) = delete;

                    friend void tag_invoke(
                        start_t, operation_state& os) noexcept
                    {
                        pika::detail::try_catch_exception_ptr(
                            [&]() {
                                pika::execution::experimental::set_value(
                                    PIKA_MOVE(os.receiver),
                                    PIKA_MOVE(os.ts).template get<Is>()...);
                            },
                            [&](std::exception_ptr ep) {
                                pika::execution::experimental::set_error(
                                    PIKA_MOVE(os.receiver), PIKA_MOVE(ep));
                            });
                    }
                };

                template <typename Receiver>
                friend auto tag_invoke(
                    connect_t, just_sender_type&& s, Receiver&& receiver)
                {
                    return operation_state<Receiver>{
                        PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.ts)};
                }

                template <typename Receiver>
                friend auto tag_invoke(
                    connect_t, just_sender_type& s, Receiver&& receiver)
                {
                    return operation_state<Receiver>{
                        PIKA_FORWARD(Receiver, receiver), s.ts};
                }
            };
        };

        template <typename Is, typename... Ts>
        using just_sender =
            typename just_sender_impl<Is, Ts...>::just_sender_type;
    }    // namespace just_detail

    inline constexpr struct just_t final
    {
        template <typename... Ts>
        constexpr PIKA_FORCEINLINE auto operator()(Ts&&... ts) const
        {
            return just_detail::just_sender<
                typename pika::util::make_index_pack<sizeof...(Ts)>::type,
                Ts...>{PIKA_FORWARD(Ts, ts)...};
        }
    } just{};
}}}    // namespace pika::execution::experimental
#endif
