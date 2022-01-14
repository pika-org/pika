//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/type_support/pack.hpp>
#include <pika/type_support/void_guard.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Tuple>
        struct fused_index_pack
          : make_index_pack<
                pika::tuple_size<typename std::decay<Tuple>::type>::value>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Tuple, typename Is>
        struct invoke_fused_result_impl;

        template <typename F, typename Tuple, std::size_t... Is>
        struct invoke_fused_result_impl<F, Tuple&, index_pack<Is...>>
          : util::invoke_result<F,
                typename pika::tuple_element<Is, Tuple>::type&...>
        {
        };

        template <typename F, typename Tuple, std::size_t... Is>
        struct invoke_fused_result_impl<F, Tuple&&, index_pack<Is...>>
          : util::invoke_result<F,
                typename pika::tuple_element<Is, Tuple>::type&&...>
        {
        };

        template <typename F, typename Tuple>
        struct invoke_fused_result
          : invoke_fused_result_impl<F, Tuple&&,
                typename fused_index_pack<Tuple>::type>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <std::size_t... Is, typename F, typename Tuple>
        constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE
            typename invoke_fused_result<F, Tuple>::type
            invoke_fused_impl(index_pack<Is...>, F&& f, Tuple&& t)
        {
            return PIKA_INVOKE(
                PIKA_FORWARD(F, f), pika::get<Is>(PIKA_FORWARD(Tuple, t))...);
        }
    }    // namespace detail

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
    constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE
        typename detail::invoke_fused_result<F, Tuple>::type
        invoke_fused(F&& f, Tuple&& t)
    {
        using index_pack = typename detail::fused_index_pack<Tuple>::type;
        return detail::invoke_fused_impl(
            index_pack{}, PIKA_FORWARD(F, f), PIKA_FORWARD(Tuple, t));
    }

    /// \copydoc invoke_fused
    ///
    /// \tparam R The result type of the function when it's called
    ///           with the content of the given sequenced type.
    template <typename R, typename F, typename Tuple>
    constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE R invoke_fused_r(F&& f, Tuple&& t)
    {
        using index_pack = typename detail::fused_index_pack<Tuple>::type;
        return util::void_guard<R>(),
               detail::invoke_fused_impl(
                   index_pack{}, PIKA_FORWARD(F, f), PIKA_FORWARD(Tuple, t));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace functional {
        struct invoke_fused
        {
            template <typename F, typename Tuple>
            constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE
                typename util::detail::invoke_fused_result<F, Tuple>::type
                operator()(F&& f, Tuple&& t) const
            {
                using index_pack =
                    typename util::detail::fused_index_pack<Tuple>::type;
                return util::detail::invoke_fused_impl(
                    index_pack{}, PIKA_FORWARD(F, f), PIKA_FORWARD(Tuple, t));
            }
        };

        template <typename R>
        struct invoke_fused_r
        {
            template <typename F, typename Tuple>
            constexpr PIKA_HOST_DEVICE PIKA_FORCEINLINE R operator()(
                F&& f, Tuple&& t) const
            {
                using index_pack =
                    typename util::detail::fused_index_pack<Tuple>::type;
                return util::void_guard<R>(),
                       util::detail::invoke_fused_impl(index_pack{},
                           PIKA_FORWARD(F, f), PIKA_FORWARD(Tuple, t));
            }
        };
    }    // namespace functional
    /// \endcond
}}    // namespace pika::util
