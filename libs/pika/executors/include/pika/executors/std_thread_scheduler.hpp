//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/execute.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>

#include <exception>
#include <thread>
#include <type_traits>
#include <utility>

namespace pika::execution::experimental {
    struct std_thread_scheduler
    {
        constexpr std_thread_scheduler() = default;

        /// \cond NOINTERNAL
        bool operator==(std_thread_scheduler const&) const noexcept { return true; }

        bool operator!=(std_thread_scheduler const&) const noexcept { return false; }

        template <typename F>
        friend void tag_invoke(execute_t, std_thread_scheduler const&, F&& f)
        {
            std::thread t{std::forward<F>(f)};
            t.detach();
        }

        template <typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Receiver_,
                typename = std::enable_if_t<
                    std::is_same_v<std::decay_t<Receiver_>, std::decay_t<Receiver>>>>
            operation_state(Receiver_&& receiver)
              : receiver(std::forward<Receiver_>(receiver))
            {
            }
            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            void start() & noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        std::thread t{[&]() mutable {
                            pika::execution::experimental::set_value(std::move(receiver));
                        }};
                        t.detach();
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            std::move(receiver), std::move(ep));
                    });
            }
        };

        struct sender
        {
            PIKA_STDEXEC_SENDER_CONCEPT

            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures = pika::execution::experimental::completion_signatures<
                pika::execution::experimental::set_value_t(),
                pika::execution::experimental::set_error_t(std::exception_ptr)>;

            template <typename Receiver>
            operation_state<Receiver> connect(Receiver&& receiver) const&
            {
                return {std::forward<Receiver>(receiver)};
            }

            struct env
            {
                friend constexpr std_thread_scheduler tag_invoke(
                    pika::execution::experimental::get_completion_scheduler_t<
                        pika::execution::experimental::set_value_t>,
                    env const&) noexcept
                {
                    return {};
                }
            };

            constexpr env get_env() const& noexcept { return {}; }
        };

        friend sender tag_invoke(schedule_t, std_thread_scheduler const&) { return {}; }
        /// \endcond
    };
}    // namespace pika::execution::experimental
