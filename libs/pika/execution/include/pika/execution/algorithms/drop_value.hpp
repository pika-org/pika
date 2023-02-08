//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <pika/concepts/concepts.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/completion_scheduler.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika::drop_value_detail {
    template <typename Receiver>
    struct drop_value_receiver_impl
    {
        struct drop_value_receiver_type;
    };

    template <typename Receiver>
    using drop_value_receiver =
        typename drop_value_receiver_impl<Receiver>::drop_value_receiver_type;

    template <typename Receiver>
    struct drop_value_receiver_impl<Receiver>::drop_value_receiver_type
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

        template <typename Error>
        friend void tag_invoke(pika::execution::experimental::set_error_t,
            drop_value_receiver_type&& r, Error&& error) noexcept
        {
            pika::execution::experimental::set_error(
                PIKA_MOVE(r.receiver), PIKA_FORWARD(Error, error));
        }

        friend void tag_invoke(pika::execution::experimental::set_stopped_t,
            drop_value_receiver_type&& r) noexcept
        {
            pika::execution::experimental::set_stopped(PIKA_MOVE(r.receiver));
        }

        template <typename... Ts>
        friend void tag_invoke(pika::execution::experimental::set_value_t,
            drop_value_receiver_type&& r, Ts&&...) noexcept
        {
            pika::execution::experimental::set_value(PIKA_MOVE(r.receiver));
        }

        friend constexpr pika::execution::experimental::detail::empty_env
        tag_invoke(pika::execution::experimental::get_env_t,
            drop_value_receiver_type const&) noexcept
        {
            return {};
        }
    };

    template <typename Sender>
    struct drop_value_sender_impl
    {
        struct drop_value_sender_type;
    };

    template <typename Sender>
    using drop_value_sender =
        typename drop_value_sender_impl<Sender>::drop_value_sender_type;

    template <typename Sender>
    struct drop_value_sender_impl<Sender>::drop_value_sender_type
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        template <class...>
        using empty_set_value =
            pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t()>;

        using completion_signatures =
            pika::execution::experimental::make_completion_signatures<Sender,
                pika::execution::experimental::detail::empty_env,
                pika::execution::experimental::completion_signatures<>,
                empty_set_value>;
#else
        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <typename...> class Variant>
        using error_types =
            pika::util::detail::unique_t<pika::util::detail::prepend_t<
                typename pika::execution::experimental::sender_traits<
                    Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;
#endif

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
            drop_value_sender_type const& sender)
        {
            return pika::execution::experimental::get_completion_scheduler<CPO>(
                sender.sender);
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            drop_value_sender_type&& s, Receiver&& receiver)
        {
            return pika::execution::experimental::connect(PIKA_MOVE(s.sender),
                drop_value_receiver<Receiver>{
                    PIKA_FORWARD(Receiver, receiver)});
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            drop_value_sender_type const& r, Receiver&& receiver)
        {
            return pika::execution::experimental::connect(r.sender,
                drop_value_receiver<Receiver>{
                    PIKA_FORWARD(Receiver, receiver)});
        }
    };
}    // namespace pika::drop_value_detail

namespace pika::execution::experimental {
    inline constexpr struct drop_value_t final
      : pika::functional::detail::tag_fallback<drop_value_t>
    {
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        constexpr PIKA_FORCEINLINE auto operator()(Sender&& sender) const
        {
            return drop_value_detail::drop_value_sender<Sender>{
                PIKA_FORWARD(Sender, sender)};
        }

        constexpr PIKA_FORCEINLINE auto operator()() const
        {
            return detail::partial_algorithm<drop_value_t>{};
        }
    } drop_value{};
}    // namespace pika::execution::experimental
