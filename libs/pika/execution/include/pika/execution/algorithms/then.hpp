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
#include <pika/concepts/concepts.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/completion_scheduler.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/pack.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    namespace then_detail {
        template <typename Receiver, typename F>
        struct then_receiver_impl
        {
            struct then_receiver_type;
        };

        template <typename Receiver, typename F>
        using then_receiver =
            typename then_receiver_impl<Receiver, F>::then_receiver_type;

        template <typename Receiver, typename F>
        struct then_receiver_impl<Receiver, F>::then_receiver_type
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Error>
            friend void tag_invoke(
                set_error_t, then_receiver_type&& r, Error&& error) noexcept
            {
                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.receiver), PIKA_FORWARD(Error, error));
            }

            friend void tag_invoke(
                set_stopped_t, then_receiver_type&& r) noexcept
            {
                pika::execution::experimental::set_stopped(
                    PIKA_MOVE(r.receiver));
            }

        private:
            template <typename... Ts>
            void set_value_helper(Ts&&... ts) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<pika::util::detail::
                                              invoke_result_t<F, Ts...>>)
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                            PIKA_INVOKE(std::move(f), PIKA_FORWARD(Ts, ts)...);
#else
                            PIKA_INVOKE(PIKA_MOVE(f), PIKA_FORWARD(Ts, ts)...);
#endif
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(receiver));
                        }
                        else
                        {
                        // Certain versions of GCC with optimizations fail on
                        // the move with an internal compiler error.
#if defined(PIKA_GCC_VERSION) && (PIKA_GCC_VERSION < 100000)
                            auto&& result = PIKA_INVOKE(
                                std::move(f), PIKA_FORWARD(Ts, ts)...);
#else
                            auto&& result = PIKA_INVOKE(
                                PIKA_MOVE(f), PIKA_FORWARD(Ts, ts)...);
#endif
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(receiver), PIKA_MOVE(result));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            PIKA_MOVE(receiver), PIKA_MOVE(ep));
                    });
            }

            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, then_receiver_type&& r, Ts&&... ts) noexcept
            {
                // GCC 7 fails with an internal compiler error unless the actual
                // body is in a helper function.
                r.set_value_helper(PIKA_FORWARD(Ts, ts)...);
            }
        };

        template <typename Sender, typename F>
        struct then_sender_impl
        {
            struct then_sender_type;
        };

        template <typename Sender, typename F>
        using then_sender =
            typename then_sender_impl<Sender, F>::then_sender_type;

        template <typename Sender, typename F>
        struct then_sender_impl<Sender, F>::then_sender_type
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type =
                    pika::util::detail::invoke_result_t<F, Ts...>;
                using type =
                    std::conditional_t<std::is_void<result_type>::value,
                        Tuple<>, Tuple<result_type>>;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                pika::util::detail::unique_t<pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template value_types<Tuple, Variant>,
                    invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types =
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename CPO,
                // clang-format off
                PIKA_CONCEPT_REQUIRES_(
                    pika::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    pika::execution::experimental::detail::has_completion_scheduler_v<
                        CPO, std::decay_t<Sender>>)
                // clang-format on
                >
            friend constexpr auto tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<CPO>,
                then_sender_type const& sender)
            {
                return pika::execution::experimental::get_completion_scheduler<
                    CPO>(sender.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, then_sender_type&& s, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(
                    PIKA_MOVE(s.sender),
                    then_receiver<Receiver, F>{
                        PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, then_sender_type& r, Receiver&& receiver)
            {
                return pika::execution::experimental::connect(r.sender,
                    then_receiver<Receiver, F>{
                        PIKA_FORWARD(Receiver, receiver), r.f});
            }
        };
    }    // namespace then_detail

    inline constexpr struct then_t final
      : pika::functional::detail::tag_fallback<then_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            then_t, Sender&& sender, F&& f)
        {
            return then_detail::then_sender<Sender, F>{
                PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            then_t, F&& f)
        {
            return detail::partial_algorithm<then_t, F>{PIKA_FORWARD(F, f)};
        }
    } then{};
}}}    // namespace pika::execution::experimental
#endif
