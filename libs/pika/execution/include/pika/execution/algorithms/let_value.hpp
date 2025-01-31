//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#else
# include <pika/assert.hpp>
# include <pika/concepts/concepts.hpp>
# include <pika/datastructures/variant.hpp>
# include <pika/errors/try_catch_exception_ptr.hpp>
# include <pika/execution/algorithms/detail/partial_algorithm.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/execution_base/sender.hpp>
# include <pika/functional/detail/tag_fallback_invoke.hpp>
# include <pika/type_support/detail/with_result_of.hpp>
# include <pika/type_support/pack.hpp>

# include <exception>
# include <tuple>
# include <type_traits>
# include <utility>

namespace pika::let_value_detail {
    template <typename PredecessorSender, typename F>
    struct let_value_sender_impl
    {
        struct let_value_sender_type;
    };

    template <typename PredecessorSender, typename F>
    using let_value_sender =
        typename let_value_sender_impl<PredecessorSender, F>::let_value_sender_type;

    template <typename PredecessorSender, typename F>
    struct let_value_sender_impl<PredecessorSender, F>::let_value_sender_type
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<PredecessorSender> predecessor_sender;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

        template <typename Tuple>
        struct predecessor_value_types_helper
        {
            using type = pika::util::detail::transform_t<Tuple, std::decay>;
        };

        // Type of the potential values returned from the predecessor sender
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using predecessor_value_types = pika::util::detail::transform_t<
            typename pika::execution::experimental::sender_traits<
                PredecessorSender>::template value_types<Tuple, Variant>,
            predecessor_value_types_helper>;

        template <typename Tuple>
        struct successor_sender_types_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct successor_sender_types_helper<Tuple<Ts...>>
        {
            using type = std::invoke_result_t<F, std::add_lvalue_reference_t<Ts>...>;
            static_assert(pika::execution::experimental::is_sender<std::decay_t<type>>::value,
                "let_value expects the invocable sender factory to return a sender");
        };

        // Type of the potential senders returned from the sender factory F
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using successor_sender_types = pika::util::detail::unique_t<pika::util::detail::transform_t<
            predecessor_value_types<Tuple, Variant>, successor_sender_types_helper>>;

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = pika::util::detail::unique_t<pika::util::detail::concat_pack_of_packs_t<
            pika::util::detail::transform_t<successor_sender_types<Tuple, Variant>,
                pika::execution::experimental::detail::value_types<Tuple,
                    Variant>::template apply>>>;

        // pika::util::detail::pack acts as a concrete type in place of Tuple. It is
        // required for computing successor_sender_types, but disappears
        // from the final error_types.
        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<pika::util::detail::prepend_t<
            pika::util::detail::concat_pack_of_packs_t<pika::util::detail::transform_t<
                successor_sender_types<pika::util::detail::pack, Variant>,
                pika::execution::experimental::detail::error_types<Variant>::template apply>>,
            std::exception_ptr>>;

        static constexpr bool sends_done = false;

        template <typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            struct let_value_predecessor_receiver;

            // Type of the operation state returned when connecting the
            // predecessor sender to the let_value_predecessor_receiver
            using predecessor_operation_state_type =
                std::decay_t<pika::execution::experimental::connect_result_t<PredecessorSender&&,
                    let_value_predecessor_receiver>>;

            // Type of the potential operation states returned when
            // connecting a sender in successor_sender_types to the receiver
            // connected to the let_value_sender
            template <typename Sender>
            struct successor_operation_state_types_helper
            {
                using type = pika::execution::experimental::connect_result_t<Sender, Receiver>;
            };
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using successor_operation_state_types =
                pika::util::detail::transform_t<successor_sender_types<Tuple, Variant>,
                    successor_operation_state_types_helper>;

            // Operation state from connecting predecessor sender to
            // let_value_predecessor_receiver
            predecessor_operation_state_type predecessor_op_state;

            using predecessor_ts_type = pika::util::detail::prepend_t<
                predecessor_value_types<std::tuple, pika::detail::variant>,
                pika::detail::monostate>;

            // Potential values returned from the predecessor sender
            predecessor_ts_type predecessor_ts;

            // Potential operation states returned when connecting a sender
            // in successor_sender_types to the receiver connected to the
            // let_value_sender
            pika::util::detail::prepend_t<
                successor_operation_state_types<std::tuple, pika::detail::variant>,
                pika::detail::monostate>
                successor_op_state;

            struct let_value_predecessor_receiver
            {
                operation_state& op_state;

                let_value_predecessor_receiver(operation_state& op_state)
                  : op_state(op_state)
                {
                }

                template <typename Error>
                friend void tag_invoke(pika::execution::experimental::set_error_t,
                    let_value_predecessor_receiver&& r, Error&& error) noexcept
                {
                    pika::execution::experimental::set_error(
                        std::move(r.op_state.receiver), std::forward<Error>(error));
                }

                friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                    let_value_predecessor_receiver&& r) noexcept
                {
                    pika::execution::experimental::set_stopped(std::move(r.op_state.receiver));
                };

                struct start_visitor
                {
                    [[noreturn]] void PIKA_STATIC_CALL_OPERATOR(pika::detail::monostate)
                    {
                        PIKA_UNREACHABLE;
                    }

                    template <typename OperationState_,
                        typename = std::enable_if_t<!std::is_same_v<std::decay_t<OperationState_>,
                            pika::detail::monostate>>>
                    void PIKA_STATIC_CALL_OPERATOR(OperationState_& op_state)
                    {
                        pika::execution::experimental::start(op_state);
                    }
                };

                struct set_value_visitor
                {
                    operation_state& op_state;

                    [[noreturn]] void operator()(pika::detail::monostate) const
                    {
                        PIKA_UNREACHABLE;
                    }

                    template <typename T,
                        typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, pika::detail::monostate>>>
                    void operator()(T& t)
                    {
                        using operation_state_type =
                            decltype(pika::execution::experimental::connect(
                                std::apply(std::move(op_state.f), t), std::declval<Receiver>()));

# if defined(PIKA_HAVE_CXX17_COPY_ELISION)
                        // with_result_of is used to emplace the operation state
                        // returned from connect without any intermediate copy
                        // construction (the operation state is not required to be
                        // copyable nor movable).
                        op_state.successor_op_state.template emplace<operation_state_type>(
                            pika::detail::with_result_of([&]() {
                                return pika::execution::experimental::connect(
                                    std::apply(std::move(op_state.f), t),
                                    std::move(op_state.receiver));
                            }));
# else
                        // MSVC doesn't get copy elision quite right, the operation
                        // state must be constructed explicitly directly in place
                        op_state.successor_op_state.template emplace_f<operation_state_type>(
                            pika::execution::experimental::connect,
                            std::apply(std::move(op_state.f), t), std::move(op_state.receiver));
# endif
                        pika::detail::visit(start_visitor{}, op_state.successor_op_state);
                    }
                };

                // This typedef is duplicated from the parent struct. The
                // parent typedef is not instantiated early enough for use
                // here.
                using predecessor_ts_type = pika::util::detail::prepend_t<
                    predecessor_value_types<std::tuple, pika::detail::variant>,
                    pika::detail::monostate>;

                template <typename... Ts>
                auto set_value(Ts&&... ts) && noexcept
                    -> decltype(std::declval<predecessor_ts_type>()
                                    .template emplace<std::tuple<std::decay_t<Ts>...>>(
                                        std::forward<Ts>(ts)...),
                        void())
                {
                    auto r = std::move(*this);
                    pika::detail::try_catch_exception_ptr(
                        [&]() {
                            r.op_state.predecessor_ts
                                .template emplace<std::tuple<std::decay_t<Ts>...>>(
                                    std::forward<Ts>(ts)...);
                            pika::detail::visit(
                                set_value_visitor{r.op_state}, r.op_state.predecessor_ts);
                        },
                        [&](std::exception_ptr ep) {
                            pika::execution::experimental::set_error(
                                std::move(r.op_state.receiver), std::move(ep));
                        });
                }
            };

            template <typename PredecessorSender_, typename Receiver_, typename F_>
            operation_state(PredecessorSender_&& predecessor_sender, Receiver_&& receiver, F_&& f)
              : receiver(std::forward<Receiver_>(receiver))
              , f(std::forward<F_>(f))
              , predecessor_op_state{pika::execution::experimental::connect(
                    std::forward<PredecessorSender_>(predecessor_sender),
                    let_value_predecessor_receiver(*this))}
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            void start() & noexcept { pika::execution::experimental::start(predecessor_op_state); }
        };

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t, let_value_sender_type&& s,
            Receiver&& receiver)
        {
            return operation_state<Receiver>(
                std::move(s.predecessor_sender), std::forward<Receiver>(receiver), std::move(s.f));
        }

        template <typename Receiver>
        friend auto tag_invoke(
            pika::execution::experimental::connect_t, let_value_sender_type const&, Receiver&&)
        {
            static_assert(sizeof(Receiver) == 0,
                "Are you missing a std::move? The let_value sender is not copyable and thus not "
                "l-value connectable. Make sure you are passing a non-const r-value reference of "
                "the sender.");
        }
    };
}    // namespace pika::let_value_detail

namespace pika::execution::experimental {
    inline constexpr struct let_value_t final : pika::functional::detail::tag_fallback<let_value_t>
    {
    private:
        // clang-format off
        template <typename PredecessorSender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<PredecessorSender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(let_value_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            return let_value_detail::let_value_sender<PredecessorSender, F>{
                std::forward<PredecessorSender>(predecessor_sender), std::forward<F>(f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(let_value_t, F&& f)
        {
            return detail::partial_algorithm<let_value_t, F>{std::forward<F>(f)};
        }
    } let_value{};
}    // namespace pika::execution::experimental
#endif
