//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/detail/with_result_of.hpp>
#include <pika/type_support/pack.hpp>

#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::drop_op_state_detail {
    template <typename OpState>
    struct drop_op_state_receiver_impl
    {
        struct drop_op_state_receiver_type;
    };

    template <typename OpState>
    using drop_op_state_receiver =
        typename drop_op_state_receiver_impl<OpState>::drop_op_state_receiver_type;

    template <typename OpState>
    struct drop_op_state_receiver_impl<OpState>::drop_op_state_receiver_type
    {
        using is_receiver = void;

        OpState* op_state = nullptr;

        template <typename Error>
        friend void tag_invoke(pika::execution::experimental::set_error_t,
            drop_op_state_receiver_type r, Error&& error) noexcept
        {
            PIKA_ASSERT(r.op_state != nullptr);
            PIKA_ASSERT(r.op_state->op_state.has_value());

            try
            {
                auto error_local = PIKA_FORWARD(Error, error);
                r.op_state->op_state.reset();

                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.op_state->receiver), PIKA_MOVE(error_local));
            }
            catch (...)
            {
                r.op_state->op_state.reset();

                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.op_state->receiver), std::current_exception());
            }
        }

        friend void tag_invoke(
            pika::execution::experimental::set_stopped_t, drop_op_state_receiver_type r) noexcept
        {
            PIKA_ASSERT(r.op_state != nullptr);
            PIKA_ASSERT(r.op_state->op_state.has_value());

            r.op_state->op_state.reset();

            pika::execution::experimental::set_stopped(PIKA_MOVE(r.op_state->receiver));
        };

        template <typename... Ts>
        friend void tag_invoke(pika::execution::experimental::set_value_t,
            drop_op_state_receiver_type r, Ts&&... ts) noexcept
        {
            PIKA_ASSERT(r.op_state != nullptr);
            PIKA_ASSERT(r.op_state->op_state.has_value());

            try
            {
                std::tuple<std::decay_t<Ts>...> ts_local(PIKA_FORWARD(Ts, ts)...);
                r.op_state->op_state.reset();

                std::apply(pika::util::detail::bind_front(pika::execution::experimental::set_value,
                               PIKA_MOVE(r.op_state->receiver)),
                    PIKA_MOVE(ts_local));
            }
            catch (...)
            {
                r.op_state->op_state.reset();

                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.op_state->receiver), std::current_exception());
            }
        }

        friend constexpr pika::execution::experimental::empty_env tag_invoke(
            pika::execution::experimental::get_env_t, drop_op_state_receiver_type const&) noexcept
        {
            return {};
        }
    };

    template <typename Sender, typename Receiver>
    struct drop_op_state_op_state_impl
    {
        struct drop_op_state_op_state_type;
    };

    template <typename Sender, typename Receiver>
    using drop_op_state_op_state =
        typename drop_op_state_op_state_impl<Sender, Receiver>::drop_op_state_op_state_type;

    template <typename Sender, typename Receiver>
    struct drop_op_state_op_state_impl<Sender, Receiver>::drop_op_state_op_state_type
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
        using operation_state_type = pika::execution::experimental::connect_result_t<Sender,
            drop_op_state_receiver<drop_op_state_op_state_type>>;
        std::optional<operation_state_type> op_state;

        template <typename Receiver_>
        drop_op_state_op_state_type(std::decay_t<Sender> sender, Receiver_&& receiver)
          : receiver(PIKA_FORWARD(Receiver_, receiver))
          , op_state(pika::detail::with_result_of([&]() mutable {
              return pika::execution::experimental::connect(
                  PIKA_MOVE(sender), drop_op_state_receiver<drop_op_state_op_state_type>{this});
          }))
        {
        }
        drop_op_state_op_state_type(drop_op_state_op_state_type&) = delete;
        drop_op_state_op_state_type& operator=(drop_op_state_op_state_type&) = delete;
        drop_op_state_op_state_type(drop_op_state_op_state_type const&) = delete;
        drop_op_state_op_state_type& operator=(drop_op_state_op_state_type const&) = delete;

        friend void tag_invoke(
            pika::execution::experimental::start_t, drop_op_state_op_state_type& os) noexcept
        {
            PIKA_ASSERT(os.op_state.has_value());
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            pika::execution::experimental::start(*(os.op_state));
        }
    };

    template <typename Sender>
    struct drop_op_state_sender_impl
    {
        struct drop_op_state_sender_type;
    };

    template <typename Sender>
    using drop_op_state_sender =
        typename drop_op_state_sender_impl<Sender>::drop_op_state_sender_type;

    template <typename Sender>
    struct drop_op_state_sender_impl<Sender>::drop_op_state_sender_type
    {
        using is_sender = void;

        std::decay_t<Sender> sender;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename... Ts>
        using value_types_helper = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t(std::decay_t<Ts>&&...)>;

        using completion_signatures =
            pika::execution::experimental::make_completion_signatures<std::decay_t<Sender>,
                pika::execution::experimental::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_error_t(std::exception_ptr)>,
                value_types_helper>;
#else
        template <typename Tuple>
        struct value_types_helper
        {
            using type =
                pika::util::detail::transform_t<pika::util::detail::transform_t<Tuple, std::decay>,
                    std::add_rvalue_reference>;
        };

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types =
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template value_types<Tuple, Variant>,
                value_types_helper>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<pika::util::detail::prepend_t<
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template error_types<Variant>,
                std::decay>,
            std::exception_ptr>>;

        static constexpr bool sends_done = false;
#endif

        template <typename Sender_,
            typename Enable =
                std::enable_if_t<!std::is_same_v<std::decay_t<Sender_>, drop_op_state_sender_type>>>
        explicit drop_op_state_sender_type(Sender_&& sender)
          : sender(PIKA_FORWARD(Sender_, sender))
        {
        }

        drop_op_state_sender_type(drop_op_state_sender_type const&) = default;
        drop_op_state_sender_type& operator=(drop_op_state_sender_type const&) = default;
        drop_op_state_sender_type(drop_op_state_sender_type&&) = default;
        drop_op_state_sender_type& operator=(drop_op_state_sender_type&&) = default;

        template <typename Receiver>
        friend drop_op_state_op_state<Sender, Receiver>
        tag_invoke(pika::execution::experimental::connect_t, drop_op_state_sender_type&& s,
            Receiver&& receiver)
        {
            return {PIKA_MOVE(s.sender), PIKA_FORWARD(Receiver, receiver)};
        }

        template <typename Receiver>
        friend drop_op_state_op_state<Sender, Receiver>
        tag_invoke(pika::execution::experimental::connect_t, drop_op_state_sender_type const& s,
            Receiver&& receiver)
        {
            return {s.sender, PIKA_FORWARD(Receiver, receiver)};
        }
    };
}    // namespace pika::drop_op_state_detail

namespace pika::execution::experimental {
    inline constexpr struct drop_operation_state_t final
    {
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR(Sender&& sender)
        {
            return drop_op_state_detail::drop_op_state_sender<Sender>{PIKA_FORWARD(Sender, sender)};
        }

        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR()
        {
            return detail::partial_algorithm<drop_operation_state_t>{};
        }
    } drop_operation_state{};
}    // namespace pika::execution::experimental
