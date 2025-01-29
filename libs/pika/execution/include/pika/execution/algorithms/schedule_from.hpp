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
# include <pika/concepts/concepts.hpp>
# include <pika/datastructures/variant.hpp>
# include <pika/execution_base/completion_scheduler.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/functional/bind_front.hpp>
# include <pika/functional/detail/invoke_result_plain_function.hpp>
# include <pika/functional/detail/tag_fallback_invoke.hpp>
# include <pika/type_support/detail/with_result_of.hpp>
# include <pika/type_support/pack.hpp>

# include <atomic>
# include <cstddef>
# include <exception>
# include <optional>
# include <tuple>
# include <type_traits>
# include <utility>

namespace pika::schedule_from_detail {
    template <typename Sender, typename Scheduler>
    struct schedule_from_sender_impl
    {
        struct schedule_from_sender_type;
    };

    template <typename Sender, typename Scheduler>
    using schedule_from_sender =
        typename schedule_from_sender_impl<Sender, Scheduler>::schedule_from_sender_type;

    template <typename Sender, typename Scheduler>
    struct schedule_from_sender_impl<Sender, Scheduler>::schedule_from_sender_type
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> predecessor_sender;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = typename pika::execution::experimental::sender_traits<
            Sender>::template value_types<Tuple, Variant>;

        template <template <typename...> class Variant>
        using predecessor_sender_error_types =
            typename pika::execution::experimental::sender_traits<Sender>::template error_types<
                Variant>;

        using scheduler_sender_type =
            pika::detail::invoke_result_plain_function_t<pika::execution::experimental::schedule_t,
                Scheduler>;
        template <template <typename...> class Variant>
        using scheduler_sender_error_types = typename pika::execution::experimental::sender_traits<
            scheduler_sender_type>::template error_types<Variant>;

        template <template <typename...> class Variant>
        using error_types =
            pika::util::detail::unique_concat_t<predecessor_sender_error_types<Variant>,
                scheduler_sender_error_types<Variant>>;

        static constexpr bool sends_done = false;

        template <typename Env>
        struct env
        {
            PIKA_NO_UNIQUE_ADDRESS Env e;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

            friend std::decay_t<Scheduler> tag_invoke(
                pika::execution::experimental::get_completion_scheduler_t<
                    pika::execution::experimental::set_value_t>,
                env const& e) noexcept
            {
                return e.scheduler;
            }

            template <typename Tag, PIKA_CONCEPT_REQUIRES_(std::is_invocable_v<Tag, env const&>)>
            friend decltype(auto) tag_invoke(Tag, env const& e)
            {
                return Tag{}(e.e);
            }
        };

        friend auto tag_invoke(
            pika::execution::experimental::get_env_t, schedule_from_sender_type const& s) noexcept
        {
            auto e = pika::execution::experimental::get_env(s.predecessor_sender);
            return env<std::decay_t<decltype(e)>>{std::move(e), s.scheduler};
        }

        template <typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            struct predecessor_sender_receiver;
            struct scheduler_sender_receiver;

            template <typename Tuple>
            struct value_types_helper
            {
                using type = pika::util::detail::transform_t<Tuple, std::decay>;
            };

            using value_type = pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<
                        Sender>::template value_types<std::tuple, pika::detail::variant>,
                    value_types_helper>,
                pika::detail::monostate>;
            value_type ts;

            using sender_operation_state_type =
                pika::execution::experimental::connect_result_t<Sender,
                    predecessor_sender_receiver>;
            sender_operation_state_type sender_os;

            using scheduler_operation_state_type = pika::execution::experimental::connect_result_t<
                pika::detail::invoke_result_plain_function_t<
                    pika::execution::experimental::schedule_t, Scheduler>,
                scheduler_sender_receiver>;
            std::optional<scheduler_operation_state_type> scheduler_op_state;

            template <typename Sender_, typename Scheduler_, typename Receiver_>
            operation_state(
                Sender_&& predecessor_sender, Scheduler_&& scheduler, Receiver_&& receiver)
              : scheduler(std::forward<Scheduler_>(scheduler))
              , receiver(std::forward<Receiver_>(receiver))
              , sender_os(pika::execution::experimental::connect(
                    std::forward<Sender_>(predecessor_sender), predecessor_sender_receiver{*this}))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            struct predecessor_sender_receiver
            {
                operation_state& op_state;

                template <typename Error>
                friend void tag_invoke(pika::execution::experimental::set_error_t,
                    predecessor_sender_receiver&& r, Error&& error) noexcept
                {
                    r.op_state.set_error_predecessor_sender(std::forward<Error>(error));
                }

                friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                    predecessor_sender_receiver&& r) noexcept
                {
                    r.op_state.set_stopped_predecessor_sender();
                }

                // These typedefs are duplicated from the parent struct. The
                // parent typedefs are not instantiated early enough for use
                // here.
                template <typename Tuple>
                struct value_types_helper
                {
                    using type = pika::util::detail::transform_t<Tuple, std::decay>;
                };

                using value_type = pika::util::detail::prepend_t<
                    pika::util::detail::transform_t<
                        typename pika::execution::experimental::sender_traits<
                            Sender>::template value_types<std::tuple, pika::detail::variant>,
                        value_types_helper>,
                    pika::detail::monostate>;

                template <typename... Ts>
                auto set_value(Ts&&... ts) && noexcept
                    -> decltype(std::declval<value_type>()
                                    .template emplace<std::tuple<std::decay_t<Ts>...>>(
                                        std::forward<Ts>(ts)...),
                        void())
                {
                    auto r = std::move(*this);
                    r.op_state.set_value_predecessor_sender(std::forward<Ts>(ts)...);
                }
            };

            template <typename Error>
            void set_error_predecessor_sender(Error&& error) noexcept
            {
                pika::execution::experimental::set_error(
                    std::move(receiver), std::forward<Error>(error));
            }

            void set_stopped_predecessor_sender() noexcept
            {
                pika::execution::experimental::set_stopped(std::move(receiver));
            }

            template <typename... Us>
            void set_value_predecessor_sender(Us&&... us) noexcept
            {
                ts.template emplace<std::tuple<std::decay_t<Us>...>>(std::forward<Us>(us)...);
# if defined(PIKA_HAVE_CXX17_COPY_ELISION)
                // with_result_of is used to emplace the operation
                // state returned from connect without any
                // intermediate copy construction (the operation
                // state is not required to be copyable nor movable).
                scheduler_op_state.emplace(pika::detail::with_result_of([&]() {
                    return pika::execution::experimental::connect(
                        pika::execution::experimental::schedule(std::move(scheduler)),
                        scheduler_sender_receiver{*this});
                }));
# else
                // MSVC doesn't get copy elision quite right, the operation
                // state must be constructed explicitly directly in place
                scheduler_op_state.emplace_f(pika::execution::experimental::connect,
                    pika::execution::experimental::schedule(std::move(scheduler)),
                    scheduler_sender_receiver{*this});
# endif
                pika::execution::experimental::start(*scheduler_op_state);
            }

            struct scheduler_sender_receiver
            {
                operation_state& op_state;

                template <typename Error>
                friend void tag_invoke(pika::execution::experimental::set_error_t,
                    scheduler_sender_receiver&& r, Error&& error) noexcept
                {
                    r.op_state.set_error_scheduler_sender(std::forward<Error>(error));
                }

                friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                    scheduler_sender_receiver&& r) noexcept
                {
                    r.op_state.set_stopped_scheduler_sender();
                }

                void set_value() && noexcept
                {
                    auto r = std::move(*this);
                    r.op_state.set_value_scheduler_sender();
                }
            };

            struct scheduler_sender_value_visitor
            {
                PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                [[noreturn]] void operator()(pika::detail::monostate) const { PIKA_UNREACHABLE; }

                template <typename Ts,
                    typename = std::enable_if_t<
                        !std::is_same_v<std::decay_t<Ts>, pika::detail::monostate>>>
                void operator()(Ts&& ts)
                {
                    std::apply(pika::util::detail::bind_front(
                                   pika::execution::experimental::set_value, std::move(receiver)),
                        std::forward<Ts>(ts));
                }
            };

            template <typename Error>
            void set_error_scheduler_sender(Error&& error) noexcept
            {
                scheduler_op_state.reset();
                pika::execution::experimental::set_error(
                    std::move(receiver), std::forward<Error>(error));
            }

            void set_stopped_scheduler_sender() noexcept
            {
                scheduler_op_state.reset();
                pika::execution::experimental::set_stopped(std::move(receiver));
            }

            void set_value_scheduler_sender() noexcept
            {
                scheduler_op_state.reset();
                pika::detail::visit(
                    scheduler_sender_value_visitor{std::move(receiver)}, std::move(ts));
            }

            void start() & noexcept { pika::execution::experimental::start(sender_os); }
        };

        template <typename Receiver>
        friend operation_state<Receiver> tag_invoke(pika::execution::experimental::connect_t,
            schedule_from_sender_type&& s, Receiver&& receiver)
        {
            return {std::move(s.predecessor_sender), std::move(s.scheduler),
                std::forward<Receiver>(receiver)};
        }

        template <typename Receiver>
        friend operation_state<Receiver> tag_invoke(pika::execution::experimental::connect_t,
            schedule_from_sender_type const& s, Receiver&& receiver)
        {
            return {s.predecessor_sender, s.scheduler, std::forward<Receiver>(receiver)};
        }
    };
}    // namespace pika::schedule_from_detail

namespace pika::execution::experimental {
    PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct schedule_from_t final
      : pika::functional::detail::tag_fallback<schedule_from_t>
    {
    private:
        // clang-format off
        template <typename Scheduler, typename Sender,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(schedule_from_t, Scheduler&& scheduler, Sender&& predecessor_sender)
        {
            return schedule_from_detail::schedule_from_sender<Sender, Scheduler>{
                std::forward<Sender>(predecessor_sender), std::forward<Scheduler>(scheduler)};
        }
    } schedule_from{};
}    // namespace pika::execution::experimental
#endif
