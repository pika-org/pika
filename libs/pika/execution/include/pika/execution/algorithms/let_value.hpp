//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/tuple.hpp>
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
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika { namespace execution { namespace experimental {
    namespace detail {
        template <typename PredecessorSender, typename F>
        struct let_value_sender
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<PredecessorSender>
                predecessor_sender;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            // Type of the potential values returned from the predecessor sender
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using predecessor_value_types =
                typename pika::execution::experimental::sender_traits<
                    std::decay_t<PredecessorSender>>::
                    template value_types<Tuple, Variant>;

            template <typename Tuple>
            struct successor_sender_types_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct successor_sender_types_helper<Tuple<Ts...>>
            {
                using type = pika::util::invoke_result_t<F,
                    std::add_lvalue_reference_t<Ts>...>;
                static_assert(pika::execution::experimental::is_sender<
                                  std::decay_t<type>>::value,
                    "let_value expects the invocable sender factory to return "
                    "a sender");
            };

            // Type of the potential senders returned from the sender factor F
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using successor_sender_types =
                pika::util::detail::unique_t<pika::util::detail::transform_t<
                    predecessor_value_types<Tuple, Variant>,
                    successor_sender_types_helper>>;

            // The workaround for clang is due to a parsing bug in clang < 11
            // in CUDA mode (where >>> also has a different meaning in kernel
            // launches).
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = pika::util::detail::unique_t<
                pika::util::detail::concat_pack_of_packs_t<pika::util::detail::
                        transform_t<successor_sender_types<Tuple, Variant>,
                            value_types<Tuple, Variant>::template apply
#if defined(PIKA_CLANG_VERSION) && PIKA_CLANG_VERSION < 110000
                            >
                    //
                    >>;
#else
                            >>>;
#endif

            // pika::util::pack acts as a concrete type in place of Tuple. It is
            // required for computing successor_sender_types, but disappears
            // from the final error_types.
            template <template <typename...> class Variant>
            using error_types =
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    pika::util::detail::concat_pack_of_packs_t<
                        pika::util::detail::transform_t<
                            successor_sender_types<pika::util::pack, Variant>,
                            error_types<Variant>::template apply>>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            struct operation_state
            {
                struct let_value_predecessor_receiver;

                // Type of the operation state returned when connecting the
                // predecessor sender to the let_value_predecessor_receiver
                using predecessor_operation_state_type =
                    std::decay_t<connect_result_t<PredecessorSender&&,
                        let_value_predecessor_receiver>>;

                // Type of the potential operation states returned when
                // connecting a sender in successor_sender_types to the receiver
                // connected to the let_value_sender
                template <typename Sender>
                struct successor_operation_state_types_helper
                {
                    using type = connect_result_t<Sender, Receiver>;
                };
                template <template <typename...> class Tuple,
                    template <typename...> class Variant>
                using successor_operation_state_types =
                    pika::util::detail::transform_t<
                        successor_sender_types<Tuple, Variant>,
                        successor_operation_state_types_helper>;

                // Operation state from connecting predecessor sender to
                // let_value_predecessor_receiver
                predecessor_operation_state_type predecessor_op_state;

                using predecessor_ts_type = pika::util::detail::prepend_t<
                    predecessor_value_types<pika::tuple, pika::variant>,
                    pika::monostate>;

                // Potential values returned from the predecessor sender
                predecessor_ts_type predecessor_ts;

                // Potential operation states returned when connecting a sender
                // in successor_sender_types to the receiver connected to the
                // let_value_sender
                pika::util::detail::prepend_t<
                    successor_operation_state_types<pika::tuple, pika::variant>,
                    pika::monostate>
                    successor_op_state;

                struct let_value_predecessor_receiver
                {
                    PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                    PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                    operation_state& op_state;

                    template <typename Receiver_, typename F_>
                    let_value_predecessor_receiver(
                        Receiver_&& receiver, F_&& f, operation_state& op_state)
                      : receiver(PIKA_FORWARD(Receiver_, receiver))
                      , f(PIKA_FORWARD(F_, f))
                      , op_state(op_state)
                    {
                    }

                    template <typename Error>
                    friend void tag_invoke(set_error_t,
                        let_value_predecessor_receiver&& r,
                        Error&& error) noexcept
                    {
                        pika::execution::experimental::set_error(
                            PIKA_MOVE(r.receiver), PIKA_FORWARD(Error, error));
                    }

                    friend void tag_invoke(
                        set_done_t, let_value_predecessor_receiver&& r) noexcept
                    {
                        pika::execution::experimental::set_done(
                            PIKA_MOVE(r.receiver));
                    };

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

                    struct set_value_visitor
                    {
                        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                        operation_state& op_state;

                        PIKA_NORETURN void operator()(pika::monostate) const
                        {
                            PIKA_UNREACHABLE;
                        }

                        template <typename T,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<T>, pika::monostate>>>
                        void operator()(T& t)
                        {
                            using operation_state_type =
                                decltype(pika::execution::experimental::connect(
                                    pika::util::invoke_fused(PIKA_MOVE(f), t),
                                    std::declval<Receiver>()));

#if defined(PIKA_HAVE_CXX17_COPY_ELISION)
                            // with_result_of is used to emplace the operation state
                            // returned from connect without any intermediate copy
                            // construction (the operation state is not required to be
                            // copyable nor movable).
                            op_state.successor_op_state
                                .template emplace<operation_state_type>(
                                    pika::util::detail::with_result_of([&]() {
                                        return pika::execution::experimental::
                                            connect(pika::util::invoke_fused(
                                                        PIKA_MOVE(f), t),
                                                PIKA_MOVE(receiver));
                                    }));
#else
                            // MSVC doesn't get copy elision quite right, the operation
                            // state must be constructed explicitly directly in place
                            op_state.successor_op_state
                                .template emplace_f<operation_state_type>(
                                    pika::execution::experimental::connect,
                                    pika::util::invoke_fused(PIKA_MOVE(f), t),
                                    PIKA_MOVE(receiver));
#endif
                            pika::visit(
                                start_visitor{}, op_state.successor_op_state);
                        }
                    };

                    // This typedef is duplicated from the parent struct. The
                    // parent typedef is not instantiated early enough for use
                    // here.
                    using predecessor_ts_type = pika::util::detail::prepend_t<
                        predecessor_value_types<pika::tuple, pika::variant>,
                        pika::monostate>;

                    template <typename... Ts>
                    void set_value(Ts&&... ts)
                    {
                        pika::detail::try_catch_exception_ptr(
                            [&]() {
                                op_state.predecessor_ts
                                    .template emplace<pika::tuple<Ts...>>(
                                        PIKA_FORWARD(Ts, ts)...);
                                pika::visit(set_value_visitor{PIKA_MOVE(receiver),
                                               PIKA_MOVE(f), op_state},
                                    op_state.predecessor_ts);
                            },
                            [&](std::exception_ptr ep) {
                                pika::execution::experimental::set_error(
                                    PIKA_MOVE(receiver), PIKA_MOVE(ep));
                            });
                    }

                    template <typename... Ts>
                    friend auto tag_invoke(set_value_t,
                        let_value_predecessor_receiver&& r, Ts&&... ts) noexcept
                        -> decltype(std::declval<predecessor_ts_type>()
                                        .template emplace<pika::tuple<Ts...>>(
                                            PIKA_FORWARD(Ts, ts)...),
                            void())
                    {
                        // set_value is in a member function only because of a
                        // compiler bug in GCC 7. When the body of set_value is
                        // inlined here compilation fails with an internal
                        // compiler error.
                        r.set_value(PIKA_FORWARD(Ts, ts)...);
                    }
                };

                template <typename PredecessorSender_, typename Receiver_,
                    typename F_>
                operation_state(PredecessorSender_&& predecessor_sender,
                    Receiver_&& receiver, F_&& f)
                  : predecessor_op_state{pika::execution::experimental::connect(
                        PIKA_FORWARD(PredecessorSender_, predecessor_sender),
                        let_value_predecessor_receiver(
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
                        os.predecessor_op_state);
                }
            };

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, let_value_sender&& s, Receiver&& receiver)
            {
                return operation_state<Receiver>(PIKA_MOVE(s.predecessor_sender),
                    PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f));
            }
        };
    }    // namespace detail

    inline constexpr struct let_value_t final
      : pika::functional::detail::tag_fallback<let_value_t>
    {
    private:
        // clang-format off
        template <typename PredecessorSender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<PredecessorSender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            let_value_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            return detail::let_value_sender<PredecessorSender, F>{
                PIKA_FORWARD(PredecessorSender, predecessor_sender),
                PIKA_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(
            let_value_t, F&& f)
        {
            return detail::partial_algorithm<let_value_t, F>{PIKA_FORWARD(F, f)};
        }
    } let_value{};
}}}    // namespace pika::execution::experimental
