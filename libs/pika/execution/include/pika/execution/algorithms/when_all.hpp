//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
#include <pika/execution_base/p2300_forward.hpp>
#else
#include <pika/concepts/concepts.hpp>
#include <pika/datastructures/member_pack.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/detail/single_result.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace pika::when_all_impl {
    // This is a receiver to be connected to the ith predecessor sender
    // passed to when_all. When set_value is called, it will emplace the
    // values sent into the appropriate position in the pack used to store
    // values from all predecessor senders.
    template <typename OperationState>
    struct when_all_receiver_impl
    {
        struct when_all_receiver_type;
    };

    template <typename OperationState>
    using when_all_receiver =
        typename when_all_receiver_impl<OperationState>::when_all_receiver_type;

    template <typename OperationState>
    struct when_all_receiver_impl<OperationState>::when_all_receiver_type
    {
        std::decay_t<OperationState>& op_state;

        when_all_receiver_type(std::decay_t<OperationState>& op_state)
          : op_state(op_state)
        {
        }

        template <typename Error>
        friend void tag_invoke(pika::execution::experimental::set_error_t,
            when_all_receiver_type&& r, Error&& error) noexcept
        {
            if (!r.op_state.set_stopped_error_called.exchange(true))
            {
                try
                {
                    r.op_state.error = PIKA_FORWARD(Error, error);
                }
                catch (...)
                {
                    // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                    r.op_state.error = std::current_exception();
                }
            }

            r.op_state.finish();
        }

        friend void tag_invoke(pika::execution::experimental::set_stopped_t,
            when_all_receiver_type&& r) noexcept
        {
            r.op_state.set_stopped_error_called = true;
            r.op_state.finish();
        };

        template <typename... Ts, std::size_t... Is>
        auto set_value_helper(pika::util::detail::index_pack<Is...>, Ts&&... ts)
            -> decltype((std::declval<typename OperationState::
                                 value_types_storage_type>()
                                .template get<OperationState::i_storage_offset +
                                    Is>()
                                .emplace(PIKA_FORWARD(Ts, ts)),
                            ...),
                void())
        {
            // op_state.ts holds values from all predecessor senders. We
            // emplace the values using the offset calculated while
            // constructing the operation state.
            (op_state.ts.template get<OperationState::i_storage_offset + Is>()
                    .emplace(PIKA_FORWARD(Ts, ts)),
                ...);
        }

        using index_pack_type = typename pika::util::detail::make_index_pack<
            OperationState::sender_pack_size>::type;

        template <typename... Ts>
        auto set_value(Ts&&... ts) noexcept
            -> decltype(set_value_helper(
                            index_pack_type{}, PIKA_FORWARD(Ts, ts)...),
                void())
        {
            if constexpr (OperationState::sender_pack_size > 0)
            {
                if (!op_state.set_stopped_error_called)
                {
                    try
                    {
                        set_value_helper(
                            index_pack_type{}, PIKA_FORWARD(Ts, ts)...);
                    }
                    catch (...)
                    {
                        if (!op_state.set_stopped_error_called.exchange(true))
                        {
                            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                            op_state.error = std::current_exception();
                        }
                    }
                }
            }

            op_state.finish();
        }
    };

    // Due to what appears to be a bug in clang this is not a hidden friend
    // of when_all_receiver. The trailing decltype for SFINAE in the member
    // set_value would give an error about accessing an incomplete type, if
    // the member set_value were a hidden friend tag_invoke overload
    // instead. Note that the receiver is unconstrained. That is because
    // OperationState in when_all_receiver<OperationState> cannot be deduced
    // when when_all_receiver is an alias template. Since this is in a
    // unique namespace nothing but when_all_receiver should ever find this
    // overload.
    template <typename Receiver, typename... Ts>
    auto tag_invoke(pika::execution::experimental::set_value_t, Receiver&& r,
        Ts&&... ts) noexcept
        -> decltype(r.set_value(PIKA_FORWARD(Ts, ts)...), void())
    {
        r.set_value(PIKA_FORWARD(Ts, ts)...);
    }

    template <typename... Senders>
    struct when_all_sender_impl
    {
        struct when_all_sender_type;
    };

    template <typename... Senders>
    using when_all_sender =
        typename when_all_sender_impl<Senders...>::when_all_sender_type;

    template <typename... Senders>
    struct when_all_sender_impl<Senders...>::when_all_sender_type
    {
        using senders_type =
            pika::util::detail::member_pack_for<std::decay_t<Senders>...>;
        senders_type senders;

        template <typename... Senders_>
        explicit constexpr when_all_sender_type(Senders_&&... senders)
          : senders(
                std::piecewise_construct, PIKA_FORWARD(Senders_, senders)...)
        {
        }

        template <typename Tuple>
        struct value_types_helper
        {
            using type = pika::util::detail::transform_t<Tuple, std::decay>;
        };

        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types = pika::util::detail::transform_t<
            pika::util::detail::concat_inner_packs_t<
                pika::util::detail::concat_t<
                    typename pika::execution::experimental::sender_traits<
                        Senders>::template value_types<Tuple, Variant>...>>,
            value_types_helper>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_concat_t<
            pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<
                    Senders>::template error_types<Variant>,
                std::decay>...,
            Variant<std::exception_ptr>>;

        static constexpr bool sends_done = false;

        static constexpr std::size_t num_predecessors = sizeof...(Senders);
        static_assert(num_predecessors > 0,
            "when_all expects at least one predecessor sender");

        template <std::size_t I>
        static constexpr std::size_t sender_pack_size_at_index =
            pika::execution::experimental::detail::single_variant_t<
                typename pika::execution::experimental::sender_traits<
                    pika::util::detail::at_index_t<I, Senders...>>::
                    template value_types<pika::util::detail::pack,
                        pika::util::detail::pack>>::size;

        template <typename Receiver, typename SendersPack,
            std::size_t I = num_predecessors - 1>
        struct operation_state;

        template <typename Receiver, typename SendersPack>
        struct operation_state<Receiver, SendersPack, 0>
        {
            // The index of the sender that this operation state handles.
            static constexpr std::size_t i = 0;
            // The offset at which we start to emplace values sent by the
            // ith predecessor sender.
            static constexpr std::size_t i_storage_offset = 0;
#if !defined(PIKA_CUDA_VERSION)
            // The number of values sent by the ith predecessor sender.
            static constexpr std::size_t sender_pack_size =
                sender_pack_size_at_index<i>;
#else
            // nvcc does not like using the helper sender_pack_size_at_index
            // here and complains about incmplete types. Lifting the helper
            // explicitly in here works.
            static constexpr std::size_t sender_pack_size =
                pika::execution::experimental::detail::single_variant_t<
                    typename pika::execution::experimental::sender_traits<
                        pika::util::detail::at_index_t<i, Senders...>>::
                        template value_types<pika::util::detail::pack,
                            pika::util::detail::pack>>::size;
#endif

            // Number of predecessor senders that have not yet called any of
            // the set signals.
            std::atomic<std::size_t> predecessors_remaining = num_predecessors;

            template <typename T>
            struct add_optional
            {
                using type = std::optional<std::decay_t<T>>;
            };
            using value_types_storage_type = pika::util::detail::change_pack_t<
                pika::util::detail::member_pack_for,
                pika::util::detail::transform_t<
                    pika::util::detail::concat_pack_of_packs_t<value_types<
                        pika::util::detail::pack, pika::util::detail::pack>>,
                    add_optional>>;
            // Values sent by all predecessor senders are stored here in the
            // base-case operation state. They are stored in a
            // member_pack<optional<T0>, ..., optional<Tn>>, where T0, ...,
            // Tn are the types of the values sent by all predecessor
            // senders.
            value_types_storage_type ts;

            std::optional<error_types<pika::detail::variant>> error;
            std::atomic<bool> set_stopped_error_called{false};
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            using operation_state_type =
                std::decay_t<decltype(pika::execution::experimental::connect(
                    std::declval<SendersPack>().template get<i>(),
                    when_all_receiver<operation_state>(
                        std::declval<std::decay_t<operation_state>&>())))>;
            operation_state_type op_state;

            template <typename Receiver_, typename Senders_>
            operation_state(Receiver_&& receiver, Senders_&& senders)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , op_state(pika::execution::experimental::connect(
#if defined(PIKA_CUDA_VERSION)
                    std::forward<Senders_>(senders).template get<i>(),
#else
                    PIKA_FORWARD(Senders_, senders).template get<i>(),
#endif
                    when_all_receiver<operation_state>(*this)))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            void start() & noexcept
            {
                pika::execution::experimental::start(op_state);
            }

            template <std::size_t... Is, typename... Ts>
            void set_value_helper(pika::util::detail::member_pack<
                pika::util::detail::index_pack<Is...>, Ts...>& ts)
            {
                pika::execution::experimental::set_value(PIKA_MOVE(receiver),
                    PIKA_MOVE(*(ts.template get<Is>()))...);
            }

            void finish() noexcept
            {
                if (--predecessors_remaining == 0)
                {
                    if (!set_stopped_error_called)
                    {
                        set_value_helper(ts);
                    }
                    else if (error)
                    {
                        pika::detail::visit(
                            [this](auto&& error) {
                                pika::execution::experimental::set_error(
                                    PIKA_MOVE(receiver),
                                    PIKA_FORWARD(decltype(error), error));
                            },
                            PIKA_MOVE(error.value()));
                    }
                    else
                    {
                        pika::execution::experimental::set_stopped(
                            PIKA_MOVE(receiver));
                    }
                }
            }
        };

        template <typename Receiver, typename SendersPack, std::size_t I>
        struct operation_state : operation_state<Receiver, SendersPack, I - 1>
        {
            using base_type = operation_state<Receiver, SendersPack, I - 1>;

            // The index of the sender that this operation state handles.
            static constexpr std::size_t i = I;
            // The number of values sent by the ith predecessor sender.
            static constexpr std::size_t sender_pack_size =
                sender_pack_size_at_index<i>;
            // The offset at which we start to emplace values sent by the
            // ith predecessor sender.
            static constexpr std::size_t i_storage_offset =
                base_type::i_storage_offset + base_type::sender_pack_size;

            using operation_state_type =
                std::decay_t<decltype(pika::execution::experimental::connect(
                    PIKA_FORWARD(SendersPack, senders).template get<i>(),
                    when_all_receiver<operation_state>(
                        std::declval<std::decay_t<operation_state>&>())))>;
            operation_state_type op_state;

            template <typename Receiver_, typename SendersPack_>
            operation_state(Receiver_&& receiver, SendersPack_&& senders)
              : base_type(PIKA_FORWARD(Receiver_, receiver),
                    PIKA_FORWARD(SendersPack, senders))
              , op_state(pika::execution::experimental::connect(
#if defined(PIKA_CUDA_VERSION)
                    std::forward<SendersPack_>(senders).template get<i>(),
#else
                    PIKA_FORWARD(SendersPack_, senders).template get<i>(),
#endif
                    when_all_receiver<operation_state>(*this)))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            void start() & noexcept
            {
                base_type::start();
                pika::execution::experimental::start(op_state);
            }
        };

        template <typename Receiver, typename SendersPack>
        friend void tag_invoke(pika::execution::experimental::start_t,
            operation_state<Receiver, SendersPack, num_predecessors - 1>&
                os) noexcept
        {
            os.start();
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            when_all_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver, senders_type&&,
                num_predecessors - 1>(
                PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.senders));
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            when_all_sender_type& s, Receiver&& receiver)
        {
            return operation_state<Receiver, senders_type&,
                num_predecessors - 1>(receiver, s.senders);
        }
    };
}    // namespace pika::when_all_impl

namespace pika::execution::experimental {
    inline constexpr struct when_all_t final
      : pika::functional::detail::tag_fallback<when_all_t>
    {
    private:
        // clang-format off
        template <typename... Senders,
            PIKA_CONCEPT_REQUIRES_(
                pika::util::detail::all_of_v<is_sender<Senders>...>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto
        tag_fallback_invoke(when_all_t, Senders&&... senders)
        {
            return pika::when_all_impl::when_all_sender<Senders...>{
                PIKA_FORWARD(Senders, senders)...};
        }
    } when_all{};
}    // namespace pika::execution::experimental
#endif
