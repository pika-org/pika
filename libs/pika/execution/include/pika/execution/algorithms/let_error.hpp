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
# include <type_traits>
# include <utility>

namespace pika::let_error_detail {
    template <typename PredecessorSender, typename F>
    struct type_helper
    {
        // Type of the potential values returned from the predecessor sender
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using predecessor_value_types = typename pika::execution::experimental::sender_traits<
            PredecessorSender>::template value_types<Tuple, Variant>;

        // Type of the potential errors returned from the predecessor sender
        template <template <typename...> class Variant>
        using predecessor_error_types =
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                PredecessorSender>::template error_types<Variant>,
                std::decay>;
        static_assert(!std::is_same<predecessor_error_types<pika::util::detail::pack>,
                          pika::util::detail::pack<>>::value,
            "let_error used with a predecessor that has an empty error_types. Is let_error "
            "misplaced?");

        template <typename Error>
        struct successor_sender_types_helper
        {
            using type = std::invoke_result_t<F, std::add_lvalue_reference_t<Error>>;
            static_assert(pika::execution::experimental::is_sender<std::decay_t<type>>::value,
                "let_error expects the invocable sender factory to return a sender");
        };

        template <template <typename...> class Variant>
        using successor_sender_types = pika::util::detail::unique_t<pika::util::detail::transform_t<
            predecessor_error_types<Variant>, successor_sender_types_helper>>;
    };

    template <typename PredecessorSender, typename Receiver, typename F>
    struct operation_state
    {
        PIKA_NO_UNIQUE_ADDRESS Receiver receiver;
        PIKA_NO_UNIQUE_ADDRESS F f;

        using types = type_helper<PredecessorSender, F>;

        struct let_error_predecessor_receiver
        {
            operation_state& op_state;

            let_error_predecessor_receiver(operation_state& op_state)
              : op_state(op_state)
            {
            }

            struct start_visitor
            {
                [[noreturn]] void PIKA_STATIC_CALL_OPERATOR(pika::detail::monostate)
                {
                    PIKA_UNREACHABLE;
                }

                template <typename OperationState_,
                    typename = std::enable_if_t<
                        !std::is_same_v<std::decay_t<OperationState_>, pika::detail::monostate>>>
                void PIKA_STATIC_CALL_OPERATOR(OperationState_& op_state)
                {
                    pika::execution::experimental::start(op_state);
                }
            };

            struct set_error_visitor
            {
                operation_state& op_state;

                [[noreturn]] void operator()(pika::detail::monostate) const { PIKA_UNREACHABLE; }

                template <typename Error,
                    typename = std::enable_if_t<
                        !std::is_same_v<std::decay_t<Error>, pika::detail::monostate>>>
                void operator()(Error& error)
                {
                    using operation_state_type = decltype(pika::execution::experimental::connect(
                        PIKA_INVOKE(std::move(op_state.f), error), std::declval<Receiver>()));

# if defined(PIKA_HAVE_CXX17_COPY_ELISION)
                    // with_result_of is used to emplace the operation
                    // state returned from connect without any
                    // intermediate copy construction (the operation
                    // state is not required to be copyable nor movable).
                    op_state.successor_op_state.template emplace<operation_state_type>(
                        pika::detail::with_result_of([&]() {
                            return pika::execution::experimental::connect(
                                PIKA_INVOKE(std::move(op_state.f), error),
                                std::move(op_state.receiver));
                        }));
# else
                    // MSVC doesn't get copy elision quite right, the operation
                    // state must be constructed explicitly directly in place
                    op_state.successor_op_state.template emplace_f<operation_state_type>(
                        pika::execution::experimental::connect,
                        PIKA_INVOKE(std::move(op_state.f), error), std::move(op_state.receiver));
# endif
                    pika::detail::visit(start_visitor{}, op_state.successor_op_state);
                }
            };

            template <typename Error>
            friend void tag_invoke(pika::execution::experimental::set_error_t,
                let_error_predecessor_receiver&& r, Error&& error) noexcept
            {
                pika::detail::try_catch_exception_ptr(
                    [&]() {
                        r.op_state.predecessor_error.template emplace<std::decay_t<Error>>(
                            std::forward<Error>(error));
                        pika::detail::visit(
                            set_error_visitor{r.op_state}, r.op_state.predecessor_error);
                    },
                    [&](std::exception_ptr ep) {
                        pika::execution::experimental::set_error(
                            std::move(r.op_state.receiver), std::move(ep));
                    });
            }

            friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                let_error_predecessor_receiver&& r) noexcept
            {
                pika::execution::experimental::set_stopped(std::move(r.op_state.receiver));
            };

            template <typename... Ts,
                typename = std::enable_if_t<std::is_invocable_v<
                    pika::execution::experimental::set_value_t, Receiver&&, Ts...>>>
            void set_value(Ts&&... ts) && noexcept
            {
                auto r = std::move(*this);
                pika::execution::experimental::set_value(
                    std::move(r.op_state.receiver), std::forward<Ts>(ts)...);
            }
        };

        // Type of the operation state returned when connecting the
        // predecessor sender to the let_error_predecessor_receiver
        using predecessor_operation_state_type =
            std::decay_t<pika::execution::experimental::connect_result_t<PredecessorSender&&,
                let_error_predecessor_receiver>>;

        // Type of the potential operation states returned when
        // connecting a sender in successor_sender_types to the receiver
        // connected to the let_error_sender
        template <typename Sender>
        struct successor_operation_state_types_helper
        {
            using type = pika::execution::experimental::connect_result_t<Sender, Receiver>;
        };
        template <template <typename...> class Variant>
        using successor_operation_state_types = pika::util::detail::transform_t<
            typename types::template successor_sender_types<Variant>,
            successor_operation_state_types_helper>;

        // Operation state from connecting predecessor sender to
        // let_error_predecessor_receiver
        predecessor_operation_state_type predecessor_operation_state;

        // Potential errors returned from the predecessor sender
        pika::util::detail::prepend_t<
            typename types::template predecessor_error_types<pika::detail::variant>,
            pika::detail::monostate>
            predecessor_error;

        // Potential operation states returned when connecting a sender
        // in successor_sender_types to the receiver connected to the
        // let_error_sender
        pika::util::detail::prepend_t<successor_operation_state_types<pika::detail::variant>,
            pika::detail::monostate>
            successor_op_state;

        template <typename PredecessorSender_, typename Receiver_, typename F_>
        operation_state(PredecessorSender_&& predecessor_sender, Receiver_&& receiver, F_&& f)
          : receiver(std::forward<Receiver_>(receiver))
          , f(std::forward<F_>(f))
          , predecessor_operation_state{pika::execution::experimental::connect(
                std::forward<PredecessorSender_>(predecessor_sender),
                let_error_predecessor_receiver(*this))}
        {
        }

        operation_state(operation_state&&) = delete;
        operation_state& operator=(operation_state&&) = delete;
        operation_state(operation_state const&) = delete;
        operation_state& operator=(operation_state const&) = delete;

        void start() & noexcept
        {
            pika::execution::experimental::start(predecessor_operation_state);
        }
    };

    template <typename PredecessorSender, typename F>
    struct let_error_sender
    {
        PIKA_NO_UNIQUE_ADDRESS PredecessorSender predecessor_sender;
        PIKA_NO_UNIQUE_ADDRESS F f;

        using types = type_helper<PredecessorSender, F>;

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = pika::util::detail::unique_concat_t<
            typename types::template predecessor_value_types<Tuple, Variant>,
            pika::util::detail::concat_pack_of_packs_t<pika::util::detail::transform_t<
                typename types::template successor_sender_types<Variant>,
                pika::execution::experimental::detail::value_types<Tuple,
                    Variant>::template apply>>>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<pika::util::detail::prepend_t<
            pika::util::detail::concat_pack_of_packs_t<pika::util::detail::transform_t<
                typename types::template successor_sender_types<Variant>,
                pika::execution::experimental::detail::error_types<Variant>::template apply>>,
            std::exception_ptr>>;

        static constexpr bool sends_done = false;

        template <typename Receiver>
        auto connect(Receiver&& receiver) &&
        {
            return operation_state<PredecessorSender, std::decay_t<Receiver>, F>(
                std::move(predecessor_sender), std::forward<Receiver>(receiver), std::move(f));
        }

        template <typename Receiver>
        auto connect(Receiver&&) const&
        {
            static_assert(sizeof(Receiver) == 0,
                "Are you missing a std::move? The let_error sender is not copyable and thus not "
                "l-value connectable. Make sure you are passing a non-const r-value reference of "
                "the sender.");
        }
    };
}    // namespace pika::let_error_detail

namespace pika::execution::experimental {
    inline constexpr struct let_error_t final : pika::functional::detail::tag_fallback<let_error_t>
    {
    private:
        // clang-format off
        template <typename PredecessorSender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                is_sender_v<PredecessorSender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(let_error_t, PredecessorSender&& predecessor_sender, F&& f)
        {
            return let_error_detail::let_error_sender<std::decay_t<PredecessorSender>,
                std::decay_t<F>>{
                std::forward<PredecessorSender>(predecessor_sender), std::forward<F>(f)};
        }

        template <typename F>
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(let_error_t, F&& f)
        {
            return detail::partial_algorithm<let_error_t, F>{std::forward<F>(f)};
        }
    } let_error{};
}    // namespace pika::execution::experimental
#endif
