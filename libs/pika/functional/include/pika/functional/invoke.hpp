//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/functional/detail/invoke.hpp>
#include <pika/type_support/void_guard.hpp>
#endif

#include <type_traits>
#include <utility>

namespace pika::util::detail {
#define PIKA_INVOKE_R(R, F, ...) (::pika::detail::void_guard<R>(), PIKA_INVOKE(F, __VA_ARGS__))

    /// Invokes the given callable object f with the content of
    /// the argument pack vs
    ///
    /// \param f Requires to be a callable object.
    ///          If f is a member function pointer, the first argument in
    ///          the pack will be treated as the callee (this object).
    ///
    /// \param vs An arbitrary pack of arguments
    ///
    /// \returns The result of the callable object when it's called with
    ///          the given argument types.
    ///
    /// \throws std::exception like objects thrown by call to object f
    ///         with the argument types vs.
    ///
    /// \note This function is similar to `std::invoke` (C++17)
    template <typename F, typename... Ts>
    constexpr PIKA_HOST_DEVICE std::invoke_result_t<F, Ts...> invoke(F&& f, Ts&&... vs)
    {
        return PIKA_INVOKE(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \copydoc invoke
    ///
    /// \tparam R The result type of the function when it's called
    ///           with the content of the given argument types vs.
    template <typename R, typename F, typename... Ts>
    constexpr PIKA_HOST_DEVICE R invoke_r(F&& f, Ts&&... vs)
    {
        return PIKA_INVOKE_R(R, PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, vs)...);
    }
}    // namespace pika::util::detail
