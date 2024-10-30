//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/errors/error_code.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/logging.hpp>
#include <pika/type_support/detail/with_result_of.hpp>
#include <pika/type_support/pack.hpp>

#if __has_include(<fmt/base.h>)
# include <fmt/base.h>
#else
# include <fmt/core.h>
#endif
#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <exception>
#include <iostream>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

// We only make the choice of mode available when not using stdexec and if not
// using older versions of GCC. stdexec's sender concepts require nothrow
// destructibility, which is not satisfied by throw_on_unstarted.  With stdexec
// enabled, an unstarted sender will always terminate. Older versions of GCC
// don't handle the noexcept(false) destructor correctly and fail at runtime.
#if !(defined(PIKA_HAVE_STDEXEC) || (defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION < 100000))
# define PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE
#endif

#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
# define PIKA_DETAIL_REQUIRE_STARTED_NOEXCEPT noexcept(false)
#else
# define PIKA_DETAIL_REQUIRE_STARTED_NOEXCEPT noexcept
#endif

namespace pika {
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
    namespace execution::experimental {
        enum class require_started_mode
        {
            terminate_on_unstarted,
            throw_on_unstarted
        };
    }    // namespace execution::experimental
#endif

    namespace require_started_detail {
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
        using pika::execution::experimental::require_started_mode;

# define PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(mode, f, message)                     \
     {                                                                                             \
         switch (mode)                                                                             \
         {                                                                                         \
         case require_started_mode::terminate_on_unstarted:                                        \
             PIKA_LOG(err, "{}: {}", f, message);                                                  \
             std::terminate();                                                                     \
             break;                                                                                \
                                                                                                   \
         case require_started_mode::throw_on_unstarted:                                            \
             PIKA_THROW_EXCEPTION(pika::error::invalid_status, f, fmt::runtime(message));          \
             break;                                                                                \
         }                                                                                         \
     }
#else
# define PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(f, message)                           \
     fmt::print(std::cerr, "{}: {}\n", f, message);                                                \
     std::terminate();
#endif

        template <typename OpState>
        struct require_started_receiver_impl
        {
            struct require_started_receiver_type;
        };

        template <typename OpState>
        using require_started_receiver =
            typename require_started_receiver_impl<OpState>::require_started_receiver_type;

        template <typename OpState>
        struct require_started_receiver_impl<OpState>::require_started_receiver_type
        {
            PIKA_STDEXEC_RECEIVER_CONCEPT

            OpState* op_state = nullptr;

            template <typename Error>
            friend void tag_invoke(pika::execution::experimental::set_error_t,
                require_started_receiver_type r, Error&& error) noexcept
            {
                PIKA_ASSERT(r.op_state != nullptr);
                pika::execution::experimental::set_error(
                    PIKA_MOVE(r.op_state->receiver), PIKA_FORWARD(Error, error));
            }

            friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                require_started_receiver_type r) noexcept
            {
                PIKA_ASSERT(r.op_state != nullptr);
                pika::execution::experimental::set_stopped(PIKA_MOVE(r.op_state->receiver));
            };

            template <typename... Ts>
            friend void tag_invoke(pika::execution::experimental::set_value_t,
                require_started_receiver_type r, Ts&&... ts) noexcept
            {
                PIKA_ASSERT(r.op_state != nullptr);
                pika::execution::experimental::set_value(
                    PIKA_MOVE(r.op_state->receiver), PIKA_FORWARD(Ts, ts)...);
            }

            friend constexpr pika::execution::experimental::empty_env tag_invoke(
                pika::execution::experimental::get_env_t,
                require_started_receiver_type const&) noexcept
            {
                return {};
            }
        };

        template <typename Sender, typename Receiver>
        struct require_started_op_state_impl
        {
            struct require_started_op_state_type;
        };

        template <typename Sender, typename Receiver>
        using require_started_op_state =
            typename require_started_op_state_impl<Sender, Receiver>::require_started_op_state_type;

        template <typename Sender, typename Receiver>
        struct require_started_op_state_impl<Sender, Receiver>::require_started_op_state_type
        {
            using operation_state_type = pika::execution::experimental::connect_result_t<Sender,
                require_started_receiver<require_started_op_state_type>>;

            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            std::optional<operation_state_type> op_state{std::nullopt};
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
            require_started_mode mode{require_started_mode::terminate_on_unstarted};
#endif
            bool started{false};

            template <typename Receiver_>
            require_started_op_state_type(std::decay_t<Sender> sender, Receiver_&& receiver
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                ,
                require_started_mode mode
#endif
                )
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , op_state(pika::detail::with_result_of([&]() {
                  return pika::execution::experimental::connect(PIKA_MOVE(sender),
                      require_started_receiver<require_started_op_state_type>{this});
              }))
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
              , mode(mode)
#endif
            {
            }

            ~require_started_op_state_type() PIKA_DETAIL_REQUIRE_STARTED_NOEXCEPT
            {
                if (!started)
                {
                    op_state.reset();

#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(mode,
                        "pika::execution::experimental::~require_started_operation_state",
                        "The operation state of a require_started sender was never started");
#else
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(
                        "pika::execution::experimental::~require_started_operation_state",
                        "The operation state of a require_started sender was never started");
#endif
                }
            }
            require_started_op_state_type(require_started_op_state_type&) = delete;
            require_started_op_state_type& operator=(require_started_op_state_type&) = delete;
            require_started_op_state_type(require_started_op_state_type const&) = delete;
            require_started_op_state_type& operator=(require_started_op_state_type const&) = delete;

            friend void tag_invoke(
                pika::execution::experimental::start_t, require_started_op_state_type& os) noexcept
            {
                PIKA_ASSERT(os.op_state.has_value());

                os.started = true;

                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                pika::execution::experimental::start(*(os.op_state));
            }
        };

        template <typename Sender>
        struct require_started_sender_impl
        {
            struct require_started_sender_type;
        };

        template <typename Sender>
        using require_started_sender =
            typename require_started_sender_impl<Sender>::require_started_sender_type;

        template <typename Sender>
        struct require_started_sender_impl<Sender>::require_started_sender_type
        {
            PIKA_STDEXEC_SENDER_CONCEPT

            std::optional<std::decay_t<Sender>> sender{std::nullopt};
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
            require_started_mode mode{require_started_mode::terminate_on_unstarted};
#endif
            mutable bool connected{false};

#if defined(PIKA_HAVE_STDEXEC)
            using completion_signatures =
                pika::execution::experimental::transform_completion_signatures_of<
                    std::decay_t<Sender>, pika::execution::experimental::empty_env>;
#else
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = typename pika::execution::experimental::sender_traits<
                Sender>::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using error_types = typename pika::execution::experimental::sender_traits<
                Sender>::template error_types<Variant>;

            static constexpr bool sends_done = false;
#endif

            template <typename Sender_,
                typename Enable = std::enable_if_t<
                    !std::is_same_v<std::decay_t<Sender_>, require_started_sender_type>>>
            explicit require_started_sender_type(Sender_&& sender
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                ,
                require_started_mode mode = require_started_mode::terminate_on_unstarted
#endif
                )
              : sender(PIKA_FORWARD(Sender_, sender))
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
              , mode(mode)
#endif
            {
            }

            require_started_sender_type(require_started_sender_type&& other) noexcept
              : sender(std::exchange(other.sender, std::nullopt))
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
              , mode(other.mode)
#endif
              , connected(other.connected)
            {
            }

            require_started_sender_type& operator=(
                require_started_sender_type&& other) PIKA_DETAIL_REQUIRE_STARTED_NOEXCEPT
            {
                if (sender.has_value() && !connected)
                {
                    sender.reset();

#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(mode,
                        "pika::execution::experimental::require_started_sender::operator=(require_"
                        "started_sender&&)",
                        "Assigning to a require_started sender that was never started, the target "
                        "would be discarded");
#else
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(
                        "pika::execution::experimental::require_started_sender::operator=(require_"
                        "started_sender&&)",
                        "Assigning to a require_started sender that was never started, the target "
                        "would be discarded");
#endif
                }

                sender = std::exchange(other.sender, std::nullopt);
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                mode = other.mode;
#endif
                connected = other.connected;

                return *this;
            }

            require_started_sender_type(require_started_sender_type const& other)
              : sender(other.sender)
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
              , mode(other.mode)
#endif
              , connected(false)
            {
            }

            require_started_sender_type& operator=(require_started_sender_type const& other)
            {
                if (sender.has_value() && !connected)
                {
                    sender.reset();

#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(mode,
                        "pika::execution::experimental::require_started_sender::operator=(require_"
                        "started_sender const&)",
                        "Assigning to a require_started sender that was never started, the target "
                        "would be discarded");
#else
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(
                        "pika::execution::experimental::require_started_sender::operator=(require_"
                        "started_sender const&)",
                        "Assigning to a require_started sender that was never started, the target "
                        "would be discarded");
#endif
                }

                sender = other.sender;
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                mode = other.mode;
#endif
                connected = false;

                return *this;
            }

            ~require_started_sender_type() PIKA_DETAIL_REQUIRE_STARTED_NOEXCEPT
            {
                if (sender.has_value() && !connected)
                {
                    sender.reset();

#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(mode,
                        "pika::execution::experimental::~require_started_sender",
                        "A require_started sender was never started");
#else
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(
                        "pika::execution::experimental::~require_started_sender",
                        "A require_started sender was never started");
#endif
                }
            }

            template <typename Receiver>
            friend require_started_op_state<Sender, Receiver>
            tag_invoke(pika::execution::experimental::connect_t, require_started_sender_type&& s,
                Receiver&& receiver)
            {
                if (!s.sender.has_value())
                {
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(s.mode,
                        "pika::execution::experimental::connect(require_started_sender&&)",
                        "Trying to connect an empty require_started sender");
#else
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(
                        "pika::execution::experimental::connect(require_started_sender&&)",
                        "Trying to connect an empty require_started sender");
#endif
                }

                s.connected = true;
                return
                {
                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    *std::exchange(s.sender, std::nullopt), PIKA_FORWARD(Receiver, receiver)
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                                                                ,
                        s.mode
#endif
                };
            }

            template <typename Receiver>
            friend require_started_op_state<Sender, Receiver>
            tag_invoke(pika::execution::experimental::connect_t,
                require_started_sender_type const& s, Receiver&& receiver)
            {
                if (!s.sender.has_value())
                {
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(s.mode,
                        "pika::execution::experimental::connect(require_started_sender const&)",
                        "Trying to connect an empty require_started sender");
#else
                    PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER(
                        "pika::execution::experimental::connect(require_started_sender const&)",
                        "Trying to connect an empty require_started sender");
#endif
                }

                s.connected = true;
                return
                {
                    *s.sender, PIKA_FORWARD(Receiver, receiver)
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
                                   ,
                        s.mode
#endif
                };
            }

            void discard() noexcept { connected = true; }
#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
            void set_mode(require_started_mode mode) noexcept { this->mode = mode; }
#endif
        };

#undef PIKA_DETAIL_HANDLE_UNSTARTED_REQUIRE_STARTED_SENDER
    }    // namespace require_started_detail

#if defined(PIKA_DETAIL_HAVE_REQUIRE_STARTED_MODE)
# define PIKA_DETAIL_REQUIRE_STARTED_MODE_PARAMETER                                                \
     , require_started_mode mode = require_started_mode::terminate_on_unstarted
# define PIKA_DETAIL_REQUIRE_STARTED_MODE_ARGUMENT , mode
#else
# define PIKA_DETAIL_REQUIRE_STARTED_MODE_PARAMETER
# define PIKA_DETAIL_REQUIRE_STARTED_MODE_ARGUMENT
#endif

    namespace execution::experimental {
        inline constexpr struct require_started_t final
        {
            template <typename Sender, PIKA_CONCEPT_REQUIRES_(is_sender_v<Sender>)>
            constexpr PIKA_FORCEINLINE auto
            PIKA_STATIC_CALL_OPERATOR(Sender&& sender PIKA_DETAIL_REQUIRE_STARTED_MODE_PARAMETER)
            {
                return require_started_detail::require_started_sender<Sender>{
                    PIKA_FORWARD(Sender, sender) PIKA_DETAIL_REQUIRE_STARTED_MODE_ARGUMENT};
            }

            constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR()
            {
                return detail::partial_algorithm<require_started_t>{};
            }
        } require_started{};
    }    // namespace execution::experimental

#undef PIKA_DETAIL_REQUIRE_STARTED_MODE_PARAMETER
#undef PIKA_DETAIL_REQUIRE_STARTED_MODE_ARGUMENT
}    // namespace pika
