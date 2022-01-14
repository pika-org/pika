//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/type_support/detail/with_result_of.hpp>
#include <pika/type_support/pack.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    namespace detail {
        template <typename PredecessorSender, typename F>
        struct let_error_sender
        {
            PIKA_NO_UNIQUE_ADDRESS typename std::decay_t<PredecessorSender>
                predecessor_sender;
            PIKA_NO_UNIQUE_ADDRESS typename std::decay_t<F> f;

            // Type of the potential values returned from the predecessor sender
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using predecessor_value_types =
                typename pika::execution::experimental::sender_traits<
                    std::decay_t<PredecessorSender>>::
                    template value_types<Tuple, Variant>;

            // Type of the potential errors returned from the predecessor sender
            template <template <typename...> class Variant>
            using predecessor_error_types =
                typename pika::execution::experimental::sender_traits<
                    std::decay_t<PredecessorSender>>::
                    template error_types<Variant>;
            static_assert(
                !std::is_same<predecessor_error_types<pika::util::pack>,
                    pika::util::pack<>>::value,
                "let_error used with a predecessor that has an empty "
                "error_types. Is let_error misplaced?");

            template <typename Error>
            struct successor_sender_types_helper
            {
                using type = pika::util::invoke_result_t<F,
                    std::add_lvalue_reference_t<Error>>;
                static_assert(pika::execution::experimental::is_sender<
                                  std::decay_t<type>>::value,
                    "let_error expects the invocable sender factory to return "
                    "a sender");
            };

            // Type of the potential senders returned from the sender factor F
            template <template <typename...> class Variant>
            using successor_sender_types = pika::util::detail::unique_t<
                pika::util::detail::transform_t<predecessor_error_types<Variant>,
                    successor_sender_types_helper>>;

            // The workaround for clang is due to a parsing bug in clang < 11
            // in CUDA mode (where >>> also has a different meaning in kernel
            // launches).
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = pika::util::detail::unique_concat_t<
                predecessor_value_types<Tuple, Variant>,
                pika::util::detail::concat_pack_of_packs_t<pika::util::detail::
                        transform_t<successor_sender_types<Variant>,
                            value_types<Tuple, Variant>::template apply
#if defined(PIKA_CLANG_VERSION) && PIKA_CLANG_VERSION < 110000
                            >
                    //
                    >>;
#else
                            >>>;
#endif

            template <template <typename...> class Variant>
            using error_types =
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    pika::util::detail::concat_pack_of_packs_t<pika::util::
                            detail::transform_t<successor_sender_types<Variant>,
                                error_types<Variant>::template apply>>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            struct operation_state
            {
                struct let_error_predecessor_receiver
                {
                    PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                    PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                    operation_state& op_state;

                    template <typename Receiver_, typename F_>
                    let_error_predecessor_receiver(
                        Receiver_&& receiver, F_&& f, operation_state& op_state)
                      : receiver(PIKA_FORWARD(Receiver_, receiver))
                      , f(PIKA_FORWARD(F_, f))
                      , op_state(op_state)
                    {
                    }

                    struct start_visitor
                    {
                        PIKA_NORETURN void operator()(pika::monostate) const
                        {
                            PIKA_UNREACHABLE;
                        }

                        template <typename OperationState_,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<OperationState_>, pika::monostate>>>
                        void operator()(OperationState_& op_state) const
                        {
                            pika::execution::experimental::start(op_state);
                        }
                    };

                    struct set_error_visitor
                    {
                        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                        operation_state& op_state;

                        PIKA_NORETURN void operator()(pika::monostate) const
                        {
                            PIKA_UNREACHABLE;
                        }

                        template <typename Error,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<Error>, pika::monostate>>>
                        void operator()(Error& error)
                        {
                            using operation_state_type =
                                decltype(pika::execution::experimental::connect(
                                    PIKA_INVOKE(PIKA_MOVE(f), error),
                                    std::declval<Receiver>()));

#if defined(PIKA_HAVE_CXX17_COPY_ELISION)
                            // with_result_of is used to emplace the operation
                            // state returned from connect without any
                            // intermediate copy construction (the operation
                            // state is not required to be copyable nor movable).
                            op_state.successor_op_state
                                .template emplace<operation_state_type>(
                                    pika::util::detail::with_result_of([&]() {
                                        return pika::execution::experimental::
                                            connect(
                                                PIKA_INVOKE(PIKA_MOVE(f), error),
                                                PIKA_MOVE(receiver));
                                    }));
#else
                            // MSVC doesn't get copy elision quite right, the operation
                            // state must be constructed explicitly directly in place
                            op_state.successor_op_state
                                .template emplace_f<operation_state_type>(
                                    pika::execution::experimental::connect,
                                    PIKA_INVOKE(PIKA_MOVE(f), error),
                                    PIKA_MOVE(receiver));
#endif
                            pika::visit(
                                start_visitor{}, op_state.successor_op_state);
                        }
                    };

                    template <typename Error>
                    friend void tag_invoke(set_error_t,
                        let_error_predecessor_receiver&& r,
                        Error&& error) noexcept
                    {
                        pika::detail::try_catch_exception_ptr(
                            [&]() {
                                // TODO: receiver is moved before the visit, but
                                // the invoke inside the visit may throw.
                                r.op_state.predecessor_error
                                    .template emplace<Error>(
                                        PIKA_FORWARD(Error, error));
                                pika::visit(
                                    set_error_visitor{PIKA_MOVE(r.receiver),
                                        PIKA_MOVE(r.f), r.op_state},
                                    r.op_state.predecessor_error);
                            },
                            [&](std::exception_ptr ep) {
                                pika::execution::experimental::set_error(
                                    PIKA_MOVE(r.receiver), PIKA_MOVE(ep));
                            });
                    }

                    friend void tag_invoke(
                        set_done_t, let_error_predecessor_receiver&& r) noexcept
                    {
                        pika::execution::experimental::set_done(
                            PIKA_MOVE(r.receiver));
                    };

                    template <typename... Ts,
                        typename = std::enable_if_t<pika::is_invocable_v<
                            pika::execution::experimental::set_value_t,
                            Receiver&&, Ts...>>>
                    friend void tag_invoke(set_value_t,
                        let_error_predecessor_receiver&& r, Ts&&... ts) noexcept
                    {
                        pika::execution::experimental::set_value(
                            PIKA_MOVE(r.receiver), PIKA_FORWARD(Ts, ts)...);
                    }
                };

                // Type of the operation state returned when connecting the
                // predecessor sender to the let_error_predecessor_receiver
                using predecessor_operation_state_type =
                    std::decay_t<connect_result_t<PredecessorSender&&,
                        let_error_predecessor_receiver>>;

                // Type of the potential operation states returned when
                // connecting a sender in successor_sender_types to the receiver
                // connected to the let_error_sender
                template <typename Sender>
                struct successor_operation_state_types_helper
                {
                    using type = connect_result_t<Sender, Receiver>;
                };
                template <template <typename...> class Variant>
                using successor_operation_state_types =
                    pika::util::detail::transform_t<
                        successor_sender_types<Variant>,
                        successor_operation_state_types_helper>;

                // Operation state from connecting predecessor sender to
                // let_error_predecessor_receiver
                predecessor_operation_state_type predecessor_operation_state;

                // Potential errors returned from the predecessor sender
                pika::util::detail::prepend_t<
                    predecessor_error_types<pika::variant>, pika::monostate>
                    predecessor_error;

                // Potential operation states returned when connecting a sender
                // in successor_sender_types to the receiver connected to the
                // let_error_sender
                pika::util::detail::prepend_t<
                    successor_operation_state_types<pika::variant>,
                    pika::monostate>
                    successor_op_state;

                template <typename PredecessorSender_, typename Receiver_,
                    typename F_>
                operation_state(PredecessorSender_&& predecessor_sender,
                    Receiver_&& receiver, F_&& f)
                  : predecessor_operation_state{
                        pika::execution::experimental::connect(
                            std::forward<PredecessorSender_>(
                                predecessor_sender),
                            let_error_predecessor_receiver(
                                PIKA_FORWARD(Receiver_, receiver),
                                PIKA_FORWARD(F_, f), *this))}
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(start_t, operation_state& os) noexcept
                {
                    pika::execution::experimental::start(
                        os.predecessor_operation_state);
                }
            };

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, let_error_sender&& s, Receiver&& receiver)
            {
                return operation_state<Receiver>(PIKA_MOVE(s.predecessor_sender),
                    PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f));
            }
        };
    }    // namespace detail

    inline constexpr struct let_error_t final
      : pika::functional::detail::tag_fallback<let_error_t>
    {
    private:
        // clang-format off
        template <typename PredecessorSender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<PredecessorSender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            let_error_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            return detail::let_error_sender<PredecessorSender, F>{
                PIKA_FORWARD(PredecessorSender, predecessor_sender),
                PIKA_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            let_error_t, F&& f)
        {
            return detail::partial_algorithm<let_error_t, F>{PIKA_FORWARD(F, f)};
        }
    } let_error{};
}}}    // namespace pika::execution::experimental
