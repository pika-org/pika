//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>
#endif

#include <pika/concepts/concepts.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/synchronization/counting_semaphore.hpp>
#include <pika/type_support/pack.hpp>

#include <atomic>
#include <exception>
#include <type_traits>
#include <utility>

namespace pika::sync_wait_detail {
    struct sync_wait_error_visitor
    {
        void PIKA_STATIC_CALL_OPERATOR(std::exception_ptr ep) { std::rethrow_exception(ep); }

        template <typename Error>
        void PIKA_STATIC_CALL_OPERATOR(Error& error)
        {
            throw error;
        }
    };

    template <typename Sender>
    struct sync_wait_receiver_impl
    {
        struct sync_wait_receiver_type;
    };

    template <typename Sender>
    using sync_wait_receiver = typename sync_wait_receiver_impl<Sender>::sync_wait_receiver_type;

    template <typename Sender>
    struct sync_wait_receiver_impl<Sender>::sync_wait_receiver_type
    {
        PIKA_STDEXEC_RECEIVER_CONCEPT

#if defined(PIKA_HAVE_STDEXEC)
        // value and error_types of the predecessor sender
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using predecessor_value_types =
            pika::util::detail::unique_t<pika::execution::experimental::value_types_of_t<Sender,
                pika::execution::experimental::empty_env, Tuple, Variant>>;

        template <template <typename...> class Variant>
        using predecessor_error_types = pika::execution::experimental::error_types_of_t<Sender,
            pika::execution::experimental::empty_env, Variant>;

        // The type of the single void or non-void result that we store. If
        // there are multiple variants or multiple values sync_wait will
        // fail to compile.
        using result_type = std::decay_t<pika::execution::experimental::detail::single_result_t<
            predecessor_value_types<pika::util::detail::decayed_pack, pika::util::detail::pack>>>;

        // Constant to indicate if the type of the result from the
        // predecessor sender is void or not
        static constexpr bool is_void_result = std::is_void_v<result_type>;

        // Dummy type to indicate that set_value with void has been called
        struct void_value_type
        {
        };

        // The type of the value to store in the variant, void_value_type if
        // result_type is void, or result_type if it is not
        using value_type = std::conditional_t<is_void_result, void_value_type, result_type>;

        // The type of errors to store in the variant. This in itself is a
        // variant.
        using error_type = pika::util::detail::unique_t<pika::util::detail::prepend_t<
            pika::util::detail::transform_t<predecessor_error_types<pika::detail::variant>,
                std::decay>,
            std::exception_ptr>>;
#else
        // value and error_types of the predecessor sender
        template <template <typename...> class Tuple, template <typename...> class Variant>
        using predecessor_value_types = typename pika::execution::experimental::sender_traits<
            Sender>::template value_types<Tuple, Variant>;

        template <template <typename...> class Variant>
        using predecessor_error_types = typename pika::execution::experimental::sender_traits<
            Sender>::template error_types<Variant>;

        // The type of the single void or non-void result that we store. If
        // there are multiple variants or multiple values sync_wait will
        // fail to compile.
        using result_type = std::decay_t<pika::execution::experimental::detail::single_result_t<
            predecessor_value_types<pika::util::detail::pack, pika::util::detail::pack>>>;

        // Constant to indicate if the type of the result from the
        // predecessor sender is void or not
        static constexpr bool is_void_result = std::is_void_v<result_type>;

        // Dummy type to indicate that set_value with void has been called
        struct void_value_type
        {
        };

        // The type of the value to store in the variant, void_value_type if
        // result_type is void, or result_type if it is not
        using value_type = std::conditional_t<is_void_result, void_value_type, result_type>;

        // The type of errors to store in the variant. This in itself is a
        // variant.
        using error_type = pika::util::detail::unique_t<pika::util::detail::prepend_t<
            pika::util::detail::transform_t<predecessor_error_types<pika::detail::variant>,
                std::decay>,
            std::exception_ptr>>;
#endif

        // We use a spinlock here to allow taking the lock on non-pika threads.
        using mutex_type = pika::concurrency::detail::spinlock;

        struct shared_state
        {
            pika::binary_semaphore<> sem{0};
            pika::detail::variant<pika::detail::monostate, error_type, value_type> value;

            void wait() { sem.acquire(); }

            auto get_value()
            {
                if (pika::detail::holds_alternative<value_type>(value))
                {
                    if constexpr (is_void_result) { return; }
                    else { return std::move(pika::detail::get<value_type>(value)); }
                }
                else if (pika::detail::holds_alternative<error_type>(value))
                {
                    pika::detail::visit(
                        sync_wait_error_visitor{}, pika::detail::get<error_type>(value));
                }

                // If the variant holds a pika::detail::monostate something has gone
                // wrong and we terminate
                PIKA_UNREACHABLE;
            }
        };

        shared_state& state;

        void signal_set_called() noexcept { state.sem.release(); }

        template <typename Error>
        void set_error(Error&& error) && noexcept
        {
            auto r = std::move(*this);
            r.state.value.template emplace<error_type>(std::forward<Error>(error));
            r.signal_set_called();
        }

        void set_stopped() && noexcept
        {
            auto r = std::move(*this);
            r.signal_set_called();
        }

        template <typename... Us,
            typename = std::enable_if_t<(is_void_result && sizeof...(Us) == 0) ||
                (!is_void_result && sizeof...(Us) == 1)>>
        void set_value(Us&&... us) && noexcept
        {
            auto r = std::move(*this);
            r.state.value.template emplace<value_type>(std::forward<Us>(us)...);
            r.signal_set_called();
        }

        constexpr pika::execution::experimental::empty_env get_env() const& noexcept { return {}; }
    };
}    // namespace pika::sync_wait_detail

namespace pika::this_thread::experimental {
    inline constexpr struct sync_wait_t final : pika::functional::detail::tag_fallback<sync_wait_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr PIKA_FORCEINLINE auto tag_fallback_invoke(sync_wait_t, Sender&& sender)
        {
            using receiver_type = sync_wait_detail::sync_wait_receiver<Sender>;
            using state_type = typename receiver_type::shared_state;

            state_type state{};
            auto op_state = pika::execution::experimental::connect(
                std::forward<Sender>(sender), receiver_type{state});
            pika::execution::experimental::start(op_state);

            state.wait();
            return state.get_value();
        }
    } sync_wait{};
}    // namespace pika::this_thread::experimental
