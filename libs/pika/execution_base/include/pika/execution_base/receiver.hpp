//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#if defined(PIKA_HAVE_STDEXEC)
# include <pika/execution_base/stdexec_forward.hpp>

namespace pika::execution::experimental {
    template <typename Receiver>
    inline constexpr bool is_receiver_v = receiver<Receiver>;

    template <typename Receiver>
    struct is_receiver
    {
        static constexpr bool value = is_receiver_v<Receiver>;
    };

    template <typename Receiver, typename Completions>
    inline constexpr bool is_receiver_of_v = receiver_of<Receiver, Completions>;

    template <typename Receiver, typename Completions>
    struct is_receiver_of
    {
        static constexpr bool value = is_receiver_of_v<Receiver, Completions>;
    };
}    // namespace pika::execution::experimental

# if defined(PIKA_HAVE_STDEXEC_SENDER_RECEIVER_CONCEPTS)
#  define PIKA_STDEXEC_RECEIVER_CONCEPT                                                            \
      using is_receiver = void;                                                                    \
      using receiver_concept = stdexec::receiver_t;
# endif
#else
# include <pika/config/constexpr.hpp>
# include <pika/functional/tag_invoke.hpp>

# include <exception>
# include <type_traits>
# include <utility>

namespace pika::execution::experimental {

# if defined(DOXYGEN)
    template <typename R, typename... As>
    void set_value(R&& r, As&&... as);

    /// set_stopped is a customization point object. The expression
    /// `pika::execution::set_stopped(r)` is equivalent to:
    ///     * `r.set_stopped()`, if that expression is valid. If the function selected
    ///       does not signal the Receiver `r`'s done channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_stopped(r), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void set_stopped();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `pika::functional::detail::tag_invoke`.
    template <typename R>
    void set_stopped(R&& r);

    /// set_error is a customization point object. The expression
    /// `pika::execution::set_error(r, e)` is equivalent to:
    ///     * `r.set_stopped(e)`, if that expression is valid. If the function selected
    ///       does not send the error `e` the Receiver `r`'s error channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_error(r, e), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void set_error();`
    ///     * Otherwise, the expression is ill-formed.
    template <typename R, typename E>
    void set_error(R&& r, E&& e);
# endif

    /// Receiving values from asynchronous computations is handled by the `Receiver`
    /// concept. A `Receiver` needs to be able to receive an error or be marked as
    /// being canceled. As such, the Receiver concept is defined by having the
    /// following two customization points defined, which form the completion-signal
    /// operations:
    ///     * `pika::execution::experimental::set_stopped`
    ///     * `pika::execution::experimental::set_error`
    ///
    /// Those two functions denote the completion-signal operations. The Receiver
    /// contract is as follows:
    ///     * None of a Receiver's completion-signal operation shall be invoked
    ///       before `pika::execution::experimental::start` has been called on the operation
    ///       state object that was returned by connecting a Receiver to a sender
    ///       `pika::execution::experimental::connect`.
    ///     * Once `pika::execution::start` has been called on the operation
    ///       state object, exactly one of the Receiver's completion-signal operation
    ///       shall complete without an exception before the Receiver is destroyed
    ///
    /// Once one of the Receiver's completion-signal operation has been completed
    /// without throwing an exception, the Receiver contract has been satisfied.
    /// In other words: The asynchronous operation has been completed.
    ///
    /// \see pika::execution::experimental::is_receiver_of
    template <typename T, typename E = std::exception_ptr>
    struct is_receiver;

    /// The `receiver_of` concept is a refinement of the `Receiver` concept by
    /// requiring one additional completion-signal operation:
    ///     * `pika::execution::set_value`
    ///
    /// This completion-signal operation adds the following to the Receiver's
    /// contract:
    ///     * If `pika::execution::set_value` exits with an exception, it
    ///       is still valid to call `pika::execution::set_error` or
    ///       `pika::execution::set_stopped`
    ///
    /// \see pika::execution::traits::is_receiver
    template <typename T, typename... As>
    struct is_receiver_of;

    PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct set_value_t
    {
        template <typename Receiver, typename... Ts>
        PIKA_FORCEINLINE constexpr auto
        PIKA_STATIC_CALL_OPERATOR(Receiver&& receiver, Ts&&... ts) noexcept
            -> decltype(std::forward<Receiver>(receiver).set_value(std::forward<Ts>(ts)...))
        {
            static_assert(
                noexcept(std::forward<Receiver>(receiver).set_value(std::forward<Ts>(ts)...)),
                "std::execution receiver set_value member function must be noexcept");
            return std::forward<Receiver>(receiver).set_value(std::forward<Ts>(ts)...);
        }
    } set_value{};

    PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct set_error_t
    {
        template <typename Receiver, typename Error>
        PIKA_FORCEINLINE constexpr auto
        PIKA_STATIC_CALL_OPERATOR(Receiver&& receiver, Error&& error) noexcept
            -> decltype(std::forward<Receiver>(receiver).set_error(std::forward<Error>(error)))
        {
            static_assert(
                noexcept(std::forward<Receiver>(receiver).set_error(std::forward<Error>(error))),
                "std::execution receiver set_error member function must be noexcept");
            return std::forward<Receiver>(receiver).set_error(std::forward<Error>(error));
        }
    } set_error{};

    PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct set_stopped_t
    {
        template <typename Receiver, typename... Ts>
        PIKA_FORCEINLINE constexpr auto PIKA_STATIC_CALL_OPERATOR(Receiver&& receiver) noexcept
            -> decltype(std::forward<Receiver>(receiver).set_stopped())
        {
            static_assert(noexcept(std::forward<Receiver>(receiver).set_stopped()),
                "std::execution receiver set_stopped member function must be noexcept");
            return std::forward<Receiver>(receiver).set_stopped();
        }
    } set_stopped{};

    ///////////////////////////////////////////////////////////////////////
    namespace detail {
        template <bool ConstructionRequirements, typename T, typename E>
        struct is_receiver_impl;

        template <typename T, typename E>
        struct is_receiver_impl<false, T, E> : std::false_type
        {
        };

        template <typename T, typename E>
        struct is_receiver_impl<true, T, E>
          : std::integral_constant<bool,
                std::is_invocable_v<set_stopped_t, std::decay_t<T>&&> &&
                    std::is_invocable_v<set_error_t, std::decay_t<T>&&, E>>
        {
        };
    }    // namespace detail

    template <typename T, typename E>
    struct is_receiver
      : detail::is_receiver_impl<std::is_move_constructible<std::decay_t<T>>::value &&
                std::is_constructible<std::decay_t<T>, T>::value,
            T, E>
    {
    };

    template <typename T, typename E = std::exception_ptr>
    inline constexpr bool is_receiver_v = is_receiver<T, E>::value;

    ///////////////////////////////////////////////////////////////////////
    namespace detail {
        template <bool IsReceiverOf, typename T, typename... As>
        struct is_receiver_of_impl;

        template <typename T, typename... As>
        struct is_receiver_of_impl<false, T, As...> : std::false_type
        {
        };

        template <typename T, typename... As>
        struct is_receiver_of_impl<true, T, As...>
          : std::integral_constant<bool, std::is_invocable_v<set_value_t, std::decay_t<T>&&, As...>>
        {
        };
    }    // namespace detail

    template <typename T, typename... As>
    struct is_receiver_of : detail::is_receiver_of_impl<is_receiver_v<T>, T, As...>
    {
    };

    template <typename T, typename... As>
    inline constexpr bool is_receiver_of_v = is_receiver_of<T, As...>::value;
}    // namespace pika::execution::experimental
#endif

#if !defined(PIKA_STDEXEC_RECEIVER_CONCEPT)
# define PIKA_STDEXEC_RECEIVER_CONCEPT using is_receiver = void;
#endif
