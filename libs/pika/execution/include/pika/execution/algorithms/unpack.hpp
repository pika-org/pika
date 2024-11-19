//  Copyright (c) 2023 ETH Zurich
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
#include <pika/functional/bind_front.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::unpack_detail {
    template <typename Receiver>
    struct unpack_receiver_impl
    {
        struct unpack_receiver_type;
    };

    template <typename Receiver>
    using unpack_receiver = typename unpack_receiver_impl<Receiver>::unpack_receiver_type;

    template <typename Receiver>
    struct unpack_receiver_impl<Receiver>::unpack_receiver_type
    {
        PIKA_STDEXEC_RECEIVER_CONCEPT

        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

        template <typename Error>
        friend void tag_invoke(pika::execution::experimental::set_error_t, unpack_receiver_type&& r,
            Error&& error) noexcept
        {
            pika::execution::experimental::set_error(
                PIKA_MOVE(r.receiver), PIKA_FORWARD(Error, error));
        }

        friend void tag_invoke(
            pika::execution::experimental::set_stopped_t, unpack_receiver_type&& r) noexcept
        {
            pika::execution::experimental::set_stopped(PIKA_MOVE(r.receiver));
        }

        template <typename Ts>
        void set_value(Ts&& ts) && noexcept
        {
            auto r = PIKA_MOVE(*this);
            std::apply(pika::util::detail::bind_front(
                           pika::execution::experimental::set_value, PIKA_MOVE(r.receiver)),
                PIKA_FORWARD(Ts, ts));
        }

        friend constexpr pika::execution::experimental::empty_env tag_invoke(
            pika::execution::experimental::get_env_t, unpack_receiver_type const&) noexcept
        {
            return {};
        }
    };

#if defined(PIKA_HAVE_STDEXEC)
    template <typename IndexPack, typename T>
    struct make_value_type;

    template <typename T, std::size_t... Is>
    struct make_value_type<pika::util::detail::index_pack<Is...>, T>
    {
        using type = pika::execution::experimental::set_value_t(
            decltype(std::get<Is>(std::declval<T>()))...);
    };

    template <typename... Ts>
    struct invoke_result_helper
    {
        static_assert(sizeof...(Ts) == 0,
            "unpack expects the predecessor sender to send exactly one tuple-like type in each "
            "variant");
    };

    template <typename T>
    struct invoke_result_helper<T>
    {
        using type = typename make_value_type<
            pika::util::detail::make_index_pack_t<std::tuple_size_v<std::decay_t<T>>>, T>::type;
    };

    template <typename... Ts>
    using invoke_result_helper_t = pika::execution::experimental::completion_signatures<
        typename invoke_result_helper<Ts...>::type>;
#else
    template <typename Tuple>
    struct invoke_result_helper;

    template <template <typename...> class Tuple, typename... Ts>
    struct invoke_result_helper<Tuple<Ts...>>
    {
        static_assert(sizeof(Tuple<Ts...>) == 0,
            "unpack expects the predecessor sender to send exactly one tuple-like type in each "
            "variant");
    };

    template <typename IndexPack, template <typename...> class Tuple, typename T>
    struct make_value_type;

    template <template <typename...> class Tuple, typename T, std::size_t... Is>
    struct make_value_type<pika::util::detail::index_pack<Is...>, Tuple, T>
    {
        using type = Tuple<decltype(std::get<Is>(std::declval<T>()))...>;
    };

    template <template <typename...> class Tuple, typename T>
    struct invoke_result_helper<Tuple<T>>
    {
        using type = typename make_value_type<
            pika::util::detail::make_index_pack_t<std::tuple_size_v<std::decay_t<T>>>, Tuple,
            T>::type;
    };

    template <template <typename...> class Tuple>
    struct invoke_result_helper<Tuple<>>
    {
        using type = Tuple<>;
    };
#endif

    template <typename Sender>
    struct unpack_sender_impl
    {
        struct unpack_sender_type;
    };

    template <typename Sender>
    using unpack_sender = typename unpack_sender_impl<Sender>::unpack_sender_type;

    template <typename Sender>
    struct unpack_sender_impl<Sender>::unpack_sender_type
    {
        PIKA_STDEXEC_SENDER_CONCEPT

        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;

#if defined(PIKA_HAVE_STDEXEC)
        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<std::decay_t<Sender>,
                pika::execution::experimental::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_error_t(std::exception_ptr)>,
                invoke_result_helper_t>;
#else
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = pika::util::detail::unique_t<
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template value_types<Tuple, Variant>,
                invoke_result_helper>>;

        template <template <typename...> class Variant>
        using error_types = typename pika::execution::experimental::sender_traits<
            Sender>::template error_types<Variant>;

        static constexpr bool sends_done = false;
#endif

        template <typename Receiver>
        friend auto tag_invoke(
            pika::execution::experimental::connect_t, unpack_sender_type&& s, Receiver&& receiver)
        {
            return pika::execution::experimental::connect(
                PIKA_MOVE(s.sender), unpack_receiver<Receiver>{PIKA_FORWARD(Receiver, receiver)});
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            unpack_sender_type const& r, Receiver&& receiver)
        {
            return pika::execution::experimental::connect(
                r.sender, unpack_receiver<Receiver>{PIKA_FORWARD(Receiver, receiver)});
        }

        friend decltype(auto) tag_invoke(
            pika::execution::experimental::get_env_t, unpack_sender_type const& s) noexcept
        {
            return pika::execution::experimental::get_env(s.sender);
        }
    };
}    // namespace pika::unpack_detail

namespace pika::execution::experimental {
    struct unpack_t final : pika::functional::detail::tag_fallback<unpack_t>
    {
    private:
        template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(unpack_t, Sender&& sender)
        {
            return unpack_detail::unpack_sender<Sender>{PIKA_FORWARD(Sender, sender)};
        }

        friend constexpr PIKA_FORCEINLINE auto tag_invoke(unpack_t)
        {
            return detail::partial_algorithm<unpack_t>{};
        }
    };

    /// \brief Transforms a sender of tuples into a sender of the elements of the tuples.
    ///
    /// Sender adaptor that takes a sender of a tuple-like and returns a sender where the tuple-like
    /// has been unpacked into its elements, similarly to `std::apply`. Each completion signature
    /// must send exactly one tuple-like, not zero or more than one. The predecessor sender can have
    /// any number of completion signatures for the value channel, each sending a single tuple-like.
    /// The adaptor does not unpack tuple-likes recursively. Any type that supports the tuple
    /// protocol can be used with the adaptor.
    inline constexpr unpack_t unpack{};
}    // namespace pika::execution::experimental
