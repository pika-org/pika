//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>

namespace pika::execution::experimental {
    template <typename Scheduler>
    inline constexpr bool is_scheduler_v = scheduler<Scheduler>;

    template <typename Scheduler>
    struct is_scheduler
    {
        static constexpr bool value = is_scheduler_v<Scheduler>;
    };

    template <typename Sender>
    inline constexpr bool is_sender_v = sender<Sender>;

    template <typename Sender>
    struct is_sender
    {
        static constexpr bool value = is_sender_v<Sender>;
    };

    template <typename Sender, typename Receiver>
    inline constexpr bool is_sender_to_v = sender_to<Sender, Receiver>;

    template <typename Sender, typename Receiver>
    struct is_sender_to
    {
        static constexpr bool value = is_sender_to_v<Sender, Receiver>;
    };
}    // namespace pika::execution::experimental

# if defined(PIKA_HAVE_STDEXEC_SENDER_RECEIVER_CONCEPTS)
#  define PIKA_STDEXEC_SENDER_CONCEPT                                                              \
      using is_sender = void;                                                                      \
      using sender_concept = stdexec::sender_t;
# endif
#else
# include <pika/config/constexpr.hpp>
# include <pika/execution_base/operation_state.hpp>
# include <pika/execution_base/receiver.hpp>
# include <pika/functional/detail/invoke_result_plain_function.hpp>
# include <pika/functional/detail/tag_fallback_invoke.hpp>
# include <pika/functional/tag_invoke.hpp>
# include <pika/type_support/equality.hpp>

# include <cstddef>
# include <exception>
# include <memory>
# include <type_traits>
# include <utility>

namespace pika::execution::experimental {
# if defined(DOXYGEN)
    /// connect is a customization point object.
    /// For some subexpression `s` and `r`, let `S` be the type such that `decltype((s))`
    /// is `S` and let `R` be the type such that `decltype((r))` is `R`. The result of
    /// the expression `pika::execution::experimental::connect(s, r)` is then equivalent to:
    ///     * `s.connect(r)`, if that expression is valid and returns a type
    ///       satisfying the `operation_state`
    ///       (\see pika::execution::experimental::traits::is_operation_state)
    ///       and if `S` satisfies the `sender` concept.
    ///     * `s.connect(r)`, if that expression is valid and returns a type
    ///       satisfying the `operation_state`
    ///       (\see pika::execution::experimental::traits::is_operation_state)
    ///       and if `S` satisfies the `sender` concept.
    ///       Overload resolution is performed in a context that include the declaration
    ///       `void connect();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `pika::functional::detail::tag_invoke`.
    template <typename S, typename R>
    void connect(S&& s, R&& r);

    /// The name schedule denotes a customization point object. For some
    /// subexpression s, let S be decltype((s)). The expression schedule(s) is
    /// expression-equivalent to:
    ///
    ///     * s.schedule(), if that expression is valid and its type models
    ///       sender.
    ///     * Otherwise, schedule(s), if that expression is valid and its type
    ///       models sender with overload resolution performed in a context that
    ///       includes the declaration
    ///
    ///           void schedule();
    ///
    ///       and that does not include a declaration of schedule.
    ///
    ///      * Otherwise, schedule(s) is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `pika::functional::detail::tag_invoke`.

# endif
    // We define an empty dummy type for compatibility. Senders can define both
    // value/error_types for our non-conformant implementation, and
    // completion_signatures for the conformant implementation. Senders do not
    // have to conditionally use completion_signatures.
    template <typename... Ts>
    struct completion_signatures
    {
    };

    /// A sender is a type that is describing an asynchronous operation. The
    /// operation itself might not have started yet. In order to get the result
    /// of this asynchronous operation, a sender needs to be connected to a
    /// receiver with the corresponding value, error and done channels:
    ///     * `pika::execution::experimental::connect`
    ///
    /// In addition, `pika::execution::experimental::::sender_traits ` needs to
    /// be specialized in some form.
    ///
    /// A sender's destructor shall not block pending completion of submitted
    /// operations.
    template <typename Sender>
    struct is_sender;

    /// \see is_sender
    template <typename Sender, typename Receiver>
    struct is_sender_to;

    /// `sender_traits` expose the different value and error types exposed
    /// by a sender. This can be either specialized directly for user defined
    /// sender types or embedded value_types, error_types and sends_done
    /// inside the sender type can be provided.
    template <typename Sender>
    struct sender_traits;

    template <typename Sender>
    struct sender_traits<Sender volatile> : sender_traits<Sender>
    {
    };
    template <typename Sender>
    struct sender_traits<Sender const> : sender_traits<Sender>
    {
    };
    template <typename Sender>
    struct sender_traits<Sender&> : sender_traits<Sender>
    {
    };
    template <typename Sender>
    struct sender_traits<Sender&&> : sender_traits<Sender>
    {
    };

    namespace detail {
        template <typename Sender>
        constexpr bool specialized(...)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool specialized(
            // NOLINTNEXTLINE(bugprone-reserved-identifier)
            typename sender_traits<Sender>::__unspecialized*)
        {
            return false;
        }
    }    // namespace detail

    template <typename Sender>
    struct is_sender
      : std::integral_constant<bool,
            std::is_move_constructible<std::decay_t<Sender>>::value &&
                detail::specialized<std::decay_t<Sender>>(nullptr)>
    {
    };

    template <typename Sender>
    inline constexpr bool is_sender_v = is_sender<Sender>::value;

    struct invocable_archetype
    {
        void operator()() {}
    };

    namespace detail {
        template <typename Executor, typename F, typename Enable = void>
        struct is_executor_of_base_impl : std::false_type
        {
        };

        template <typename Executor, typename F>
        struct is_executor_of_base_impl<Executor, F,
            std::enable_if_t<std::is_invocable<std::decay_t<F>&>::value &&
                std::is_constructible<std::decay_t<F>, F>::value &&
                std::is_destructible<std::decay_t<F>>::value &&
                std::is_move_constructible<std::decay_t<F>>::value &&
                std::is_copy_constructible<Executor>::value &&
                pika::detail::is_equality_comparable_v<Executor>>> : std::true_type
        {
        };

        template <typename Executor>
        struct is_executor_base
          : is_executor_of_base_impl<std::decay_t<Executor>, invocable_archetype>
        {
        };
    }    // namespace detail

    PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct connect_t
    {
        template <typename Sender, typename Receiver>
        PIKA_FORCEINLINE constexpr auto
        PIKA_STATIC_CALL_OPERATOR(Sender&& sender, Receiver&& receiver) noexcept(
            noexcept(std::forward<Sender>(sender).connect(std::forward<Receiver>(receiver))))
            -> decltype(std::forward<Sender>(sender).connect(std::forward<Receiver>(receiver)))
        {
            return std::forward<Sender>(sender).connect(std::forward<Receiver>(receiver));
        }
    } connect{};

    namespace detail {
        template <typename F, typename E>
        struct as_receiver
        {
            F f;

            void set_value() noexcept(noexcept(f())) { f(); }

            template <typename E_>
            [[noreturn]] void set_error(E_&&) noexcept
            {
                std::terminate();
            }

            void set_stopped() noexcept {}
        };
    }    // namespace detail

    PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct schedule_t : pika::functional::detail::tag<schedule_t>
    {
    } schedule{};

    namespace detail {
        template <bool IsSenderReceiver, typename Sender, typename Receiver>
        struct is_sender_to_impl;

        template <typename Sender, typename Receiver>
        struct is_sender_to_impl<false, Sender, Receiver> : std::false_type
        {
        };

        template <typename Sender, typename Receiver>
        struct is_sender_to_impl<true, Sender, Receiver>
          : std::integral_constant<bool,
                std::is_invocable_v<connect_t, Sender&&, Receiver&&> ||
                    std::is_invocable_v<connect_t, Sender&&, Receiver&> ||
                    std::is_invocable_v<connect_t, Sender&&, Receiver const&> ||
                    std::is_invocable_v<connect_t, Sender&, Receiver&&> ||
                    std::is_invocable_v<connect_t, Sender&, Receiver&> ||
                    std::is_invocable_v<connect_t, Sender&, Receiver const&> ||
                    std::is_invocable_v<connect_t, Sender const&, Receiver&&> ||
                    std::is_invocable_v<connect_t, Sender const&, Receiver&> ||
                    std::is_invocable_v<connect_t, Sender const&, Receiver const&>>
        {
        };
    }    // namespace detail

    template <typename Sender, typename Receiver>
    struct is_sender_to
      : detail::is_sender_to_impl<is_sender_v<Sender> && is_receiver_v<Receiver>, Sender, Receiver>
    {
    };

    template <typename Sender, typename Receiver>
    inline constexpr bool is_sender_to_v = is_sender_to<Sender, Receiver>::value;

    namespace detail {
        template <typename... As>
        struct tuple_mock;
        template <typename... As>
        struct variant_mock;

        template <typename Sender>
        constexpr bool
        has_value_types(typename Sender::template value_types<tuple_mock, variant_mock>*)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool has_value_types(...)
        {
            return false;
        }

        template <typename Sender>
        constexpr bool has_error_types(typename Sender::template error_types<variant_mock>*)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool has_error_types(...)
        {
            return false;
        }

        template <typename Sender>
        constexpr bool has_sends_done(decltype(Sender::sends_done)*)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool has_sends_done(...)
        {
            return false;
        }

        template <typename Sender>
        struct has_sender_types
          : std::integral_constant<bool,
                has_value_types<Sender>(nullptr) && has_error_types<Sender>(nullptr) &&
                    has_sends_done<Sender>(nullptr)>
        {
        };

        template <bool HasSenderTraits, typename Sender>
        struct sender_traits_base;

        template <typename Sender>
        struct sender_traits_base<true /* HasSenderTraits */, Sender>
        {
            template <template <typename...> class Tuple, template <typename...> class Variant>
            using value_types = typename Sender::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using error_types = typename Sender::template error_types<Variant>;

            static constexpr bool sends_done = Sender::sends_done;
        };

        template <typename Sender>
        struct sender_traits_base<false /* HasSenderTraits */, Sender>
        {
            // NOLINTNEXTLINE(bugprone-reserved-identifier)
            using __unspecialized = void;
        };

        template <typename Sender>
        struct is_typed_sender
          : std::integral_constant<bool,
                is_sender<Sender>::value && detail::has_sender_types<Sender>::value>
        {
        };
    }    // namespace detail

    template <typename Sender>
    struct sender_traits
      : detail::sender_traits_base<detail::has_sender_types<Sender>::value, Sender>
    {
    };

    // Explicitly specialize for void to avoid forming references to void
    // (is_invocable is in the base implementation, which forms a reference to
    // the Sender type).
    template <>
    struct sender_traits<void>
    {
        // NOLINTNEXTLINE(bugprone-reserved-identifier)
        using __unspecialized = void;
    };

    namespace detail {
        template <template <typename...> class Tuple, template <typename...> class Variant>
        struct value_types
        {
            template <typename Sender>
            struct apply
            {
                using type = typename pika::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>;
            };
        };

        template <template <typename...> class Variant>
        struct error_types
        {
            template <typename Sender>
            struct apply
            {
                using type = typename pika::execution::experimental::sender_traits<
                    Sender>::template error_types<Variant>;
            };
        };
    }    // namespace detail

    template <typename Scheduler, typename Enable = void>
    struct is_scheduler : std::false_type
    {
    };

    template <typename Scheduler>
    struct is_scheduler<Scheduler,
        std::enable_if_t<std::is_invocable<schedule_t, Scheduler>::value &&
            std::is_copy_constructible<Scheduler>::value &&
            pika::detail::is_equality_comparable_v<Scheduler>>> : std::true_type
    {
    };

    template <typename Scheduler>
    inline constexpr bool is_scheduler_v = is_scheduler<Scheduler>::value;

    template <typename S, typename R>
    using connect_result_t = pika::detail::invoke_result_plain_function_t<connect_t, S, R>;

    struct empty_env
    {
    };

    template <typename T, typename Enable = void>
    struct has_get_env : std::false_type
    {
    };

    template <typename T>
    struct has_get_env<T, std::void_t<decltype(std::declval<T const&>().get_env())>>
      : std::true_type
    {
    };

    inline constexpr struct get_env_t
    {
        template <typename T>
        constexpr auto PIKA_STATIC_CALL_OPERATOR(T const& t) noexcept
        {
            if constexpr (has_get_env<T>::value)
            {
                static_assert(noexcept(t.get_env()),
                    "std::execution get_env member function must be noexcept");
                return t.get_env();
            }
            else if constexpr (is_sender_v<T>) { return empty_env{}; }
            else
            {
                static_assert(sizeof(T) == 0, "No environment for type T");
                return empty_env{};
            }
        }
    } get_env{};
}    // namespace pika::execution::experimental
#endif

namespace pika::execution::detail {
    /// Helper type for storing set_stopped signals in variants.
    struct stopped_type
    {
    };
}    // namespace pika::execution::detail

#if !defined(PIKA_STDEXEC_SENDER_CONCEPT)
# define PIKA_STDEXEC_SENDER_CONCEPT using is_sender = void;
#endif
