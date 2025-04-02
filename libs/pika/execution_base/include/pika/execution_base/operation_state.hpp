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
    template <typename OperationState>
    inline constexpr bool is_operation_state_v = operation_state<OperationState>;
}
#else
# include <pika/config/constexpr.hpp>
# include <pika/functional/tag_invoke.hpp>

# include <type_traits>
# include <utility>

namespace pika::execution::experimental {
# if defined(DOXYGEN)
    /// start is a customization point object. The expression
    /// `pika::execution::experimental::start(r)` is equivalent to:
    ///     * `r.start()`, if that expression is valid. If the function selected
    ///       does not signal the receiver `r`'s done channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `start(r), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void start();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `pika::functional::detail::tag_invoke`.
    template <typename O>
    void start(O&& o);
# endif

    /// An `operation_state` is an object representing the asynchronous operation
    /// that has been returned from calling `pika::execution::experimental::connect` with
    /// a `sender` and a `receiver`. The only operation on an `operation_state`
    /// is:
    ///     * `pika::execution::experimental::start`
    ///
    /// `pika::execution::experimental::start` can be called exactly once. Once it has
    /// been invoked, the caller needs to ensure that the receiver's completion
    /// signaling operations strongly happen before the destructor of the state
    /// is called. The call to `pika::execution::experimental::start` needs to happen
    /// strongly before the completion signaling operations.
    ///
    template <typename O>
    struct is_operation_state;

    PIKA_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct start_t
    {
        template <typename OperationState>
        PIKA_FORCEINLINE constexpr auto
        PIKA_STATIC_CALL_OPERATOR(OperationState& os) noexcept -> decltype(os.start())
        {
            static_assert(noexcept(os.start()),
                "std::execution operation state start member function must be noexcept");
            os.start();
        }
    } start{};

    namespace detail {
        template <bool IsOperationState, typename O>
        struct is_operation_state_impl;

        template <typename O>
        struct is_operation_state_impl<false, O> : std::false_type
        {
        };

        template <typename O>
        struct is_operation_state_impl<true, O>
          : std::integral_constant<bool, noexcept(start(std::declval<O&>()))>
        {
        };
    }    // namespace detail

    template <typename O>
    struct is_operation_state
      : detail::is_operation_state_impl<std::is_destructible<O>::value &&
                std::is_object<O>::value && std::is_invocable_v<start_t, std::decay_t<O>&>,
            O>
    {
    };

    template <typename O>
    inline constexpr bool is_operation_state_v = is_operation_state<O>::value;
}    // namespace pika::execution::experimental
#endif
