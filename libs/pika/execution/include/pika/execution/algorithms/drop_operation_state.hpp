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
        PIKA_STDEXEC_RECEIVER_CONCEPT

        OpState* op_state = nullptr;

        template <typename Error>
        void set_error(Error&& error) && noexcept
        {
            auto r = std::move(*this);
            PIKA_ASSERT(r.op_state != nullptr);
            PIKA_ASSERT(r.op_state->op_state.has_value());

            try
            {
                auto error_local = std::forward<Error>(error);
                r.op_state->op_state.reset();

                pika::execution::experimental::set_error(
                    std::move(r.op_state->receiver), std::move(error_local));
            }
            catch (...)
            {
                r.op_state->op_state.reset();

                pika::execution::experimental::set_error(
                    std::move(r.op_state->receiver), std::current_exception());
            }
        }

        void set_stopped() && noexcept
        {
            auto r = std::move(*this);
            PIKA_ASSERT(r.op_state != nullptr);
            PIKA_ASSERT(r.op_state->op_state.has_value());

            r.op_state->op_state.reset();

            pika::execution::experimental::set_stopped(std::move(r.op_state->receiver));
        };

        template <typename... Ts>
        void set_value(Ts&&... ts) && noexcept
        {
            auto r = std::move(*this);

            PIKA_ASSERT(r.op_state != nullptr);
            PIKA_ASSERT(r.op_state->op_state.has_value());

            try
            {
                std::tuple<std::decay_t<Ts>...> ts_local(std::forward<Ts>(ts)...);
                r.op_state->op_state.reset();

                std::apply(pika::util::detail::bind_front(pika::execution::experimental::set_value,
                               std::move(r.op_state->receiver)),
                    std::move(ts_local));
            }
            catch (...)
            {
                r.op_state->op_state.reset();

                pika::execution::experimental::set_error(
                    std::move(r.op_state->receiver), std::current_exception());
            }
        }

        constexpr pika::execution::experimental::empty_env get_env() const& noexcept { return {}; }
    };

    template <typename Sender, typename Receiver>
    struct drop_op_state_op_state
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
        using operation_state_type = pika::execution::experimental::connect_result_t<Sender,
            drop_op_state_receiver<drop_op_state_op_state>>;
        std::optional<operation_state_type> op_state;

        template <typename Receiver_>
        drop_op_state_op_state(std::decay_t<Sender> sender, Receiver_&& receiver)
          : receiver(std::forward<Receiver>(receiver))
          , op_state(pika::detail::with_result_of([&]() mutable {
              return pika::execution::experimental::connect(
                  std::move(sender), drop_op_state_receiver<drop_op_state_op_state>{this});
          }))
        {
        }
        drop_op_state_op_state(drop_op_state_op_state&) = delete;
        drop_op_state_op_state& operator=(drop_op_state_op_state&) = delete;
        drop_op_state_op_state(drop_op_state_op_state const&) = delete;
        drop_op_state_op_state& operator=(drop_op_state_op_state const&) = delete;

        void start() & noexcept
        {
            PIKA_ASSERT(op_state.has_value());
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            pika::execution::experimental::start(*(op_state));
        }
    };

    template <typename Sender>
    struct drop_op_state_sender
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        std::decay_t<Sender> sender;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename... Ts>
        using value_types_helper = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t(std::decay_t<Ts>&&...)>;

        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<std::decay_t<Sender>,
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
                std::enable_if_t<!std::is_same_v<std::decay_t<Sender_>, drop_op_state_sender>>>
        explicit drop_op_state_sender(Sender_&& sender)
          : sender(std::forward<Sender_>(sender))
        {
        }

        drop_op_state_sender(drop_op_state_sender const&) = default;
        drop_op_state_sender& operator=(drop_op_state_sender const&) = default;
        drop_op_state_sender(drop_op_state_sender&&) = default;
        drop_op_state_sender& operator=(drop_op_state_sender&&) = default;

        template <typename Receiver>
        drop_op_state_op_state<Sender, Receiver> connect(Receiver&& receiver) &&
        {
            return {std::move(sender), std::forward<Receiver>(receiver)};
        }

        template <typename Receiver>
        drop_op_state_op_state<Sender, Receiver> connect(Receiver&& receiver) const&
        {
            return {sender, std::forward<Receiver>(receiver)};
        }
    };
}    // namespace pika::drop_op_state_detail

namespace pika::execution::experimental {

    struct drop_operation_state_t final
    {
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR(Sender&& sender)
        {
            return drop_op_state_detail::drop_op_state_sender<Sender>{std::forward<Sender>(sender)};
        }

        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR()
        {
            return detail::partial_algorithm<drop_operation_state_t>{};
        }
    };

    /// \brief Releases the operation state of the adaptor before signaling a connected receiver.
    ///
    /// Sender adaptor that takes any sender and returns a sender. Values received as references
    /// from the predecessor sender will be copied before being passed on to successor senders.
    /// Other values are passed on unchanged.
    ///
    /// The operation state of previous senders can hold on to allocated memory or values longer
    /// than necessary which can prevent other algorithms from using those resources.
    /// \p drop_operation_state can be used to explicitly release the operation state, and thus
    /// associated resources, of previous senders.
    inline constexpr drop_operation_state_t drop_operation_state{};

}    // namespace pika::execution::experimental
