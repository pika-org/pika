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

namespace pika { namespace execution { namespace experimental {
    struct std_thread_scheduler
    {
        constexpr std_thread_scheduler() = default;

        /// \cond NOINTERNAL
        bool operator==(std_thread_scheduler const&) const noexcept
        {
            return true;
        }

        bool operator!=(std_thread_scheduler const&) const noexcept
        {
            return false;
        }

        template <typename F>
        friend void tag_invoke(execute_t, std_thread_scheduler const&, F&& f)
        {
            std::thread t{PIKA_FORWARD(F, f)};
            t.detach();
        }

        template <typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Receiver_,
                typename = std::enable_if_t<std::is_same_v<
                    std::decay_t<Receiver_>, std::decay_t<Receiver>>>>
            operation_state(Receiver_&& receiver)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
            {
            }
            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            friend void tag_invoke(start_t, operation_state& os) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        std::thread t{[&os]() mutable {
                            pika::execution::experimental::set_value(
                                PIKA_MOVE(os.receiver));
                        }};
                        t.detach();
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            PIKA_MOVE(os.receiver), PIKA_MOVE(ep));
                    });
            }
        };

        struct sender
        {
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            using completion_signatures =
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_value_t(),
                    pika::execution::experimental::set_error_t(
                        std::exception_ptr)>;

            template <typename Receiver>
            friend operation_state<Receiver>
            tag_invoke(connect_t, sender const&, Receiver&& receiver)
            {
                return {PIKA_FORWARD(Receiver, receiver)};
            }

            template <typename CPO,
                PIKA_CONCEPT_REQUIRES_(std::is_same_v<CPO,
                    pika::execution::experimental::set_value_t>)>
            friend constexpr std_thread_scheduler tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<CPO>,
                sender const&) noexcept
            {
                return {};
            }
        };

        friend sender tag_invoke(schedule_t, std_thread_scheduler const&)
        {
            return {};
        }
        /// \endcond
    };
}}}    // namespace pika::execution::experimental
