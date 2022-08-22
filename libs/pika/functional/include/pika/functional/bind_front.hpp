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
#include <pika/functional/invoke_result.hpp>
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
      : util::detail::invoke_result<F, Ts..., Us...>
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
            typename =
                typename std::enable_if_t<std::is_constructible_v<F, F_>>>
        constexpr explicit bound_front(F_&& f, Ts_&&... vs)
          : _f(PIKA_FORWARD(F_, f))
          , _args(std::piecewise_construct, PIKA_FORWARD(Ts_, vs)...)
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
          : _f(PIKA_MOVE(other._f))
          , _args(PIKA_MOVE(other._args))
        {
        }
#endif

        bound_front& operator=(bound_front const&) = delete;

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE typename invoke_bound_front_result<F&,
            util::detail::pack<Ts&...>, Us&&...>::type
        operator()(Us&&... vs) &
        {
            return PIKA_INVOKE(
                _f, _args.template get<Is>()..., PIKA_FORWARD(Us, vs)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE typename invoke_bound_front_result<F const&,
            util::detail::pack<Ts const&...>, Us&&...>::type
        operator()(Us&&... vs) const&
        {
            return PIKA_INVOKE(
                _f, _args.template get<Is>()..., PIKA_FORWARD(Us, vs)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE typename invoke_bound_front_result<F&&,
            util::detail::pack<Ts&&...>, Us&&...>::type
        operator()(Us&&... vs) &&
        {
            return PIKA_INVOKE(PIKA_MOVE(_f),
                PIKA_MOVE(_args).template get<Is>()...,
                PIKA_FORWARD(Us, vs)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE typename invoke_bound_front_result<F const&&,
            util::detail::pack<Ts const&&...>, Us&&...>::type
        operator()(Us&&... vs) const&&
        {
            return PIKA_INVOKE(PIKA_MOVE(_f),
                PIKA_MOVE(_args).template get<Is>()...,
                PIKA_FORWARD(Us, vs)...);
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

#if PIKA_HAVE_ITTNOTIFY != 0 && !defined(PIKA_HAVE_APEX)
        util::itt::string_handle get_function_annotation_itt() const
        {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
            return pika::detail::get_function_annotation_itt<F>::call(_f);
#else
            static util::itt::string_handle sh("bound_front");
            return sh;
#endif
        }
#endif

    private:
        F _f;
        util::detail::member_pack_for<Ts...> _args;
    };

    template <typename F, typename... Ts>
    constexpr bound_front<std::decay_t<F>,
        typename util::detail::make_index_pack<sizeof...(Ts)>::type,
        std::decay_t<Ts>...>
    bind_front(F&& f, Ts&&... vs)
    {
        using result_type = bound_front<std::decay_t<F>,
            typename util::detail::make_index_pack<sizeof...(Ts)>::type,
            std::decay_t<Ts>...>;

        return result_type(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, vs)...);
    }

    // nullary functions do not need to be bound again
    template <typename F>
    constexpr std::decay_t<F> bind_front(F&& f)
    {
        return PIKA_FORWARD(F, f);
    }
}    // namespace pika::util::detail

namespace pika::detail {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename... Ts>
    struct get_function_address<util::detail::bound_front<F, Ts...>>
    {
        static constexpr std::size_t call(
            util::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<util::detail::bound_front<F, Ts...>>
    {
        static constexpr char const* call(
            util::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if PIKA_HAVE_ITTNOTIFY != 0 && !defined(PIKA_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<util::detail::bound_front<F, Ts...>>
    {
        static util::itt::string_handle call(
            util::detail::bound_front<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
#endif
}    // namespace pika::detail
