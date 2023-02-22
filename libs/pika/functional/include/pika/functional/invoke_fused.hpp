//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/type_support/pack.hpp>
#include <pika/type_support/void_guard.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::util::detail {
    template <typename Tuple>
    struct fused_index_pack : make_index_pack<std::tuple_size_v<std::decay_t<Tuple>>>
    {
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Tuple, typename Is>
    struct invoke_fused_result_impl;

    template <typename F, typename Tuple, std::size_t... Is>
    struct invoke_fused_result_impl<F, Tuple&, index_pack<Is...>>
      : std::invoke_result<F, std::tuple_element_t<Is, Tuple>&...>
    {
    };

    template <typename F, typename Tuple, std::size_t... Is>
    struct invoke_fused_result_impl<F, Tuple&&, index_pack<Is...>>
      : std::invoke_result<F, std::tuple_element_t<Is, Tuple>&&...>
    {
    };

    template <typename F, typename Tuple>
    struct invoke_fused_result
      : invoke_fused_result_impl<F, Tuple&&, typename fused_index_pack<Tuple>::type>
    {
    };

    ///////////////////////////////////////////////////////////////////////
    template <std::size_t... Is, typename F, typename Tuple>
    constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE typename invoke_fused_result<F, Tuple>::type
    invoke_fused_impl(index_pack<Is...>, F&& f, Tuple&& t)
    {
        return PIKA_INVOKE(PIKA_FORWARD(F, f), std::get<Is>(PIKA_FORWARD(Tuple, t))...);
    }

    /// Invokes the given callable object f with the content of
    /// the sequenced type t (tuples, pairs)
    ///
    /// \param f Must be a callable object. If f is a member function pointer,
    ///          the first argument in the sequenced type will be treated as
    ///          the callee (this object).
    ///
    /// \param t A type whose contents are accessible through a call
    ///          to pika#get.
    ///
    /// \returns The result of the callable object when it's called with
    ///          the content of the given sequenced type.
    ///
    /// \throws std::exception like objects thrown by call to object f
    ///         with the arguments contained in the sequenceable type t.
    ///
    /// \note This function is similar to `std::apply` (C++17)
    template <typename F, typename Tuple>
    constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE typename detail::invoke_fused_result<F, Tuple>::type
    invoke_fused(F&& f, Tuple&& t)
    {
        using index_pack = typename detail::fused_index_pack<Tuple>::type;
        return detail::invoke_fused_impl(index_pack{}, PIKA_FORWARD(F, f), PIKA_FORWARD(Tuple, t));
    }

    /// \copydoc invoke_fused
    ///
    /// \tparam R The result type of the function when it's called
    ///           with the content of the given sequenced type.
    template <typename R, typename F, typename Tuple>
    constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE R invoke_fused_r(F&& f, Tuple&& t)
    {
        using index_pack = typename detail::fused_index_pack<Tuple>::type;
        return ::pika::detail::void_guard<R>(),
               detail::invoke_fused_impl(index_pack{}, PIKA_FORWARD(F, f), PIKA_FORWARD(Tuple, t));
    }
}    // namespace pika::util::detail
