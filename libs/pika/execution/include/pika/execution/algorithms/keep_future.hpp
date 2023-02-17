//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/acquire_shared_state.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika::keep_future_detail {
    template <typename Receiver, typename Future>
    struct operation_state
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
        std::decay_t<Future> future;

        friend void tag_invoke(pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    auto state = pika::traits::detail::get_shared_state(os.future);

                    if (!state)
                    {
                        PIKA_THROW_EXCEPTION(pika::error::no_state, "operation_state::start",
                            "the future has no valid shared state");
                    }

                    // The operation state has to be kept alive until set_value
                    // is called, which means that we don't need to move
                    // receiver and future into the on_completed callback.
                    state->set_on_completed([&os]() mutable {
                        pika::execution::experimental::set_value(
                            PIKA_MOVE(os.receiver), PIKA_MOVE(os.future));
                    });
                },
                [&](std::exception_ptr ep) {
                    pika::execution::experimental::set_error(PIKA_MOVE(os.receiver), PIKA_MOVE(ep));
                });
        }
    };

    template <typename Future>
    struct keep_future_sender_base
    {
        std::decay_t<Future> future;

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<std::decay_t<Future>>>;

        template <template <typename...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        using completion_signatures = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t(std::decay_t<Future>),
            pika::execution::experimental::set_error_t(std::exception_ptr)>;
    };

    template <typename Future>
    struct keep_future_sender;

    template <typename T>
    struct keep_future_sender<pika::future<T>> : public keep_future_sender_base<pika::future<T>>
    {
        using future_type = pika::future<T>;
        using base_type = keep_future_sender_base<pika::future<T>>;
        using base_type::future;

        template <typename Future,
            typename =
                std::enable_if_t<!std::is_same<std::decay_t<Future>, keep_future_sender>::value>>
        explicit keep_future_sender(Future&& future)
          : base_type{PIKA_FORWARD(Future, future)}
        {
        }

        keep_future_sender(keep_future_sender&&) = default;
        keep_future_sender& operator=(keep_future_sender&&) = default;
        keep_future_sender(keep_future_sender const&) = delete;
        keep_future_sender& operator=(keep_future_sender const&) = delete;

        template <typename Receiver>
        friend operation_state<Receiver, future_type> tag_invoke(
            pika::execution::experimental::connect_t, keep_future_sender&& s, Receiver&& receiver)
        {
            return {PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.future)};
        }

        template <typename Receiver>
        friend operation_state<Receiver, future_type>
        tag_invoke(pika::execution::experimental::connect_t, keep_future_sender const&, Receiver&&)
        {
            static_assert(sizeof(Receiver) == 0,
                "Are you missing a std::move? The keep_future sender of a "
                "future is not copyable (because future is not copyable) and "
                "thus not l-value connectable. Make sure you are passing an "
                "r-value reference of the sender.");
        }
    };

    template <typename T>
    struct keep_future_sender<pika::shared_future<T>>
      : keep_future_sender_base<pika::shared_future<T>>
    {
        using future_type = pika::shared_future<T>;
        using base_type = keep_future_sender_base<pika::shared_future<T>>;
        using base_type::future;

        template <typename Future,
            typename =
                std::enable_if_t<!std::is_same<std::decay_t<Future>, keep_future_sender>::value>>
        explicit keep_future_sender(Future&& future)
          : base_type{PIKA_FORWARD(Future, future)}
        {
        }

        keep_future_sender(keep_future_sender&&) = default;
        keep_future_sender& operator=(keep_future_sender&&) = default;
        keep_future_sender(keep_future_sender const&) = default;
        keep_future_sender& operator=(keep_future_sender const&) = default;

        template <typename Receiver>
        friend operation_state<Receiver, future_type> tag_invoke(
            pika::execution::experimental::connect_t, keep_future_sender&& s, Receiver&& receiver)
        {
            return {PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.future)};
        }

        template <typename Receiver>
        friend operation_state<Receiver, future_type>
        tag_invoke(pika::execution::experimental::connect_t, keep_future_sender const& s,
            Receiver&& receiver)
        {
            return {PIKA_FORWARD(Receiver, receiver), s.future};
        }
    };
}    // namespace pika::keep_future_detail

namespace pika::execution::experimental {
    inline constexpr struct keep_future_t final
    {
        // clang-format off
        template <typename Future,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_future_v<std::decay_t<Future>>
            )>
        // clang-format on
        constexpr PIKA_FORCEINLINE auto operator()(Future&& future) const
        {
            return keep_future_detail::keep_future_sender<std::decay_t<Future>>(
                PIKA_FORWARD(Future, future));
        }

        constexpr PIKA_FORCEINLINE auto operator()() const
        {
            return detail::partial_algorithm<keep_future_t>{};
        }
    } keep_future{};
}    // namespace pika::execution::experimental
