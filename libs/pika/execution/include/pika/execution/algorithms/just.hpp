//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else
# include <pika/datastructures/member_pack.hpp>
# include <pika/errors/try_catch_exception_ptr.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/type_support/pack.hpp>

# include <cstddef>
# include <exception>
# include <stdexcept>
# include <utility>

namespace pika::just_detail {
    template <typename Is, typename... Ts>
    struct just_sender_impl;

    template <typename std::size_t... Is, typename... Ts>
    struct just_sender_impl<pika::util::detail::index_pack<Is...>, Ts...>
    {
        struct just_sender_type
        {
            pika::util::detail::member_pack_for<std::decay_t<Ts>...> ts;

            constexpr just_sender_type() = default;

            template <typename T,
                typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, just_sender_type>>>
            explicit constexpr just_sender_type(T&& t)
              : ts(std::piecewise_construct, std::forward<T>(t))
            {
            }

            template <typename T0, typename T1, typename... Ts_>
            explicit constexpr just_sender_type(T0&& t0, T1&& t1, Ts_&&... ts)
              : ts(std::piecewise_construct, std::forward<T0>(t0), std::forward<T1>(t1),
                    std::forward<Ts_>(ts)...)
            {
            }

            just_sender_type(just_sender_type&&) = default;
            just_sender_type(just_sender_type const&) = default;
            just_sender_type& operator=(just_sender_type&&) = default;
            just_sender_type& operator=(just_sender_type const&) = default;

            template <template <typename...> class Tuple, template <typename...> class Variant>
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
                    pika::util::detail::member_pack_for<std::decay_t<Ts>...> ts)
                  : receiver(std::forward<Receiver_>(receiver))
                  , ts(std::move(ts))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    pika::detail::try_catch_exception_ptr(
                        [&]() {
                            pika::execution::experimental::set_value(
                                std::move(receiver), std::move(ts).template get<Is>()...);
                        },
                        [&](std::exception_ptr ep) {
                            pika::execution::experimental::set_error(
                                std::move(receiver), std::move(ep));
                        });
                }
            };

            template <typename Receiver>
            auto connect(Receiver&& receiver) &&
            {
                return operation_state<Receiver>{std::forward<Receiver>(receiver), std::move(ts)};
            }

            template <typename Receiver>
            auto connect(Receiver&& receiver) const&
            {
                return operation_state<Receiver>{std::forward<Receiver>(receiver), ts};
            }
        };
    };

    template <typename Is, typename... Ts>
    using just_sender = typename just_sender_impl<Is, Ts...>::just_sender_type;
}    // namespace pika::just_detail

namespace pika::execution::experimental {
    inline constexpr struct just_t final
    {
        template <typename... Ts>
        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR(Ts&&... ts)
        {
            return just_detail::just_sender<
                typename pika::util::detail::make_index_pack<sizeof...(Ts)>::type, Ts...>{
                std::forward<Ts>(ts)...};
        }
    } just{};
}    // namespace pika::execution::experimental
#endif
