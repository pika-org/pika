//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/datastructures/member_pack.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/one_shot.hpp>
#include <pika/functional/traits/get_function_address.hpp>
#include <pika/functional/traits/get_function_annotation.hpp>
#include <pika/type_support/decay.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika::util::detail {
    template <typename F, typename Ts, typename... Us>
    struct invoke_bound_front_result;

    template <typename F, typename... Ts, typename... Us>
    struct invoke_bound_front_result<F, util::detail::pack<Ts...>, Us...>
      : std::invoke_result<F, Ts..., Us...>
    {
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Is, typename... Ts>
    class bound_front;

    template <typename F, std::size_t... Is, typename... Ts>
    class bound_front<F, index_pack<Is...>, Ts...>
    {
    public:
        template <typename F_, typename... Ts_,
            typename = typename std::enable_if_t<std::is_constructible_v<F, F_>>>
        constexpr explicit bound_front(F_&& f, Ts_&&... vs)
          : _f(std::forward<F_>(f))
          , _args(std::piecewise_construct, std::forward<Ts_>(vs)...)
        {
        }

#if !defined(__NVCC__) && !defined(__CUDACC__)
        bound_front(bound_front const&) = default;
        bound_front(bound_front&&) = default;
#else
        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        constexpr PIKA_HOST_DEVICE bound_front(bound_front const& other)
          : _f(other._f)
          , _args(other._args)
        {
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        constexpr PIKA_HOST_DEVICE bound_front(bound_front&& other)
          : _f(std::move(other._f))
          , _args(std::move(other._args))
        {
        }
#endif

        bound_front& operator=(bound_front const&) = delete;

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE
            typename invoke_bound_front_result<F&, util::detail::pack<Ts&...>, Us&&...>::type
            operator()(Us&&... vs) &
        {
            return PIKA_INVOKE(_f, _args.template get<Is>()..., std::forward<Us>(vs)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE typename invoke_bound_front_result<F const&,
            util::detail::pack<Ts const&...>, Us&&...>::type
        operator()(Us&&... vs) const&
        {
            return PIKA_INVOKE(_f, _args.template get<Is>()..., std::forward<Us>(vs)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE
            typename invoke_bound_front_result<F&&, util::detail::pack<Ts&&...>, Us&&...>::type
            operator()(Us&&... vs) &&
        {
            return PIKA_INVOKE(
                std::move(_f), std::move(_args).template get<Is>()..., std::forward<Us>(vs)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE typename invoke_bound_front_result<F const&&,
            util::detail::pack<Ts const&&...>, Us&&...>::type
        operator()(Us&&... vs) const&&
        {
            return PIKA_INVOKE(
                std::move(_f), std::move(_args).template get<Is>()..., std::forward<Us>(vs)...);
        }

        constexpr std::size_t get_function_address() const
        {
            return pika::detail::get_function_address<F>::call(_f);
        }

        constexpr char const* get_function_annotation() const
        {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
            return pika::detail::get_function_annotation<F>::call(_f);
#else
            return nullptr;
#endif
        }

    private:
        F _f;
        util::detail::member_pack_for<Ts...> _args;
    };

    template <typename F, typename... Ts>
    constexpr bound_front<std::decay_t<F>,
        typename util::detail::make_index_pack<sizeof...(Ts)>::type, std::decay_t<Ts>...>
    bind_front(F&& f, Ts&&... vs)
    {
        using result_type = bound_front<std::decay_t<F>,
            typename util::detail::make_index_pack<sizeof...(Ts)>::type, std::decay_t<Ts>...>;

        return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    // nullary functions do not need to be bound again
    template <typename F>
    constexpr std::decay_t<F> bind_front(F&& f)
    {
        return std::forward<F>(f);
    }
}    // namespace pika::util::detail

namespace pika::detail {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename... Ts>
    struct get_function_address<pika::util::detail::bound_front<F, Ts...>>
    {
        static constexpr std::size_t call(
            pika::util::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<pika::util::detail::bound_front<F, Ts...>>
    {
        static constexpr char const* call(
            pika::util::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
#endif
}    // namespace pika::detail
