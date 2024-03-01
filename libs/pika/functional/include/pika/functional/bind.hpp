//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/datastructures/member_pack.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/one_shot.hpp>
#include <pika/functional/traits/get_function_address.hpp>
#include <pika/functional/traits/get_function_annotation.hpp>
#include <pika/functional/traits/is_bind_expression.hpp>
#include <pika/type_support/decay.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika::util::detail {
    template <std::size_t I>
    struct bind_eval_placeholder
    {
        template <typename T, typename... Us>
        static constexpr PIKA_HOST_DEVICE decltype(auto) call(T&& /*t*/, Us&&... vs)
        {
            return util::detail::member_pack_for<Us&&...>(
                std::piecewise_construct, PIKA_FORWARD(Us, vs)...)
                .template get<I>();
        }
    };

    template <typename T, std::size_t NumUs, typename TD = std::decay_t<T>, typename Enable = void>
    struct bind_eval
    {
        template <typename... Us>
        static constexpr PIKA_HOST_DEVICE T&& call(T&& t, Us&&... /*vs*/)
        {
            return PIKA_FORWARD(T, t);
        }
    };

    template <typename T, std::size_t NumUs, typename TD>
    struct bind_eval<T, NumUs, TD,
        std::enable_if_t<std::is_placeholder_v<TD> != 0 && (std::is_placeholder_v<TD> <= NumUs)>>
      : bind_eval_placeholder<(std::size_t) std::is_placeholder_v<TD> - 1>
    {
    };

    template <typename T, std::size_t NumUs, typename TD>
    struct bind_eval<T, NumUs, TD, std::enable_if_t<pika::detail::is_bind_expression_v<TD>>>
    {
        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        static constexpr PIKA_HOST_DEVICE std::invoke_result_t<T, Us...> call(T&& t, Us&&... vs)
        {
            return PIKA_INVOKE(PIKA_FORWARD(T, t), PIKA_FORWARD(Us, vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Ts, typename... Us>
    struct invoke_bound_result;

    template <typename F, typename... Ts, typename... Us>
    struct invoke_bound_result<F, util::detail::pack<Ts...>, Us...>
      : std::invoke_result<F,
            decltype(bind_eval<Ts, sizeof...(Us)>::call(
                std::declval<Ts>(), std::declval<Us>()...))...>
    {
    };

    template <typename F, typename Ts, typename... Us>
    using invoke_bound_result_t = typename invoke_bound_result<F, Ts, Us...>::type;

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Is, typename... Ts>
    class bound;

    template <typename F, std::size_t... Is, typename... Ts>
    class bound<F, index_pack<Is...>, Ts...>
    {
    public:
        template <typename F_, typename... Ts_,
            typename = std::enable_if_t<std::is_constructible_v<F, F_>>>
        constexpr explicit bound(F_&& f, Ts_&&... vs)
          : _f(PIKA_FORWARD(F_, f))
          , _args(std::piecewise_construct, PIKA_FORWARD(Ts_, vs)...)
        {
        }

#if !defined(__NVCC__) && !defined(__CUDACC__)
        bound(bound const&) = default;
        bound(bound&&) = default;
#else
        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        constexpr PIKA_HOST_DEVICE bound(bound const& other)
          : _f(other._f)
          , _args(other._args)
        {
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        constexpr PIKA_HOST_DEVICE bound(bound&& other)
          : _f(PIKA_MOVE(other._f))
          , _args(PIKA_MOVE(other._args))
        {
        }
#endif

        bound& operator=(bound const&) = delete;

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        // https://github.com/pika-org/pika/issues/993
        constexpr PIKA_NO_SANITIZE_ADDRESS
            PIKA_HOST_DEVICE invoke_bound_result_t<F&, util::detail::pack<Ts&...>, Us&&...>
            operator()(Us&&... vs) &
        {
            return PIKA_INVOKE(_f,
                detail::bind_eval<Ts&, sizeof...(Us)>::call(
                    _args.template get<Is>(), PIKA_FORWARD(Us, vs)...)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE
            invoke_bound_result_t<F const&, util::detail::pack<Ts const&...>, Us&&...>
            operator()(Us&&... vs) const&
        {
            return PIKA_INVOKE(_f,
                detail::bind_eval<Ts const&, sizeof...(Us)>::call(
                    _args.template get<Is>(), PIKA_FORWARD(Us, vs)...)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE invoke_bound_result_t<F&&, util::detail::pack<Ts&&...>, Us&&...>
        operator()(Us&&... vs) &&
        {
            return PIKA_INVOKE(PIKA_MOVE(_f),
                detail::bind_eval<Ts, sizeof...(Us)>::call(
                    PIKA_MOVE(_args).template get<Is>(), PIKA_FORWARD(Us, vs)...)...);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename... Us>
        constexpr PIKA_HOST_DEVICE
            invoke_bound_result_t<F const&&, util::detail::pack<Ts const&&...>, Us&&...>
            operator()(Us&&... vs) const&&
        {
            return PIKA_INVOKE(PIKA_MOVE(_f),
                detail::bind_eval<Ts const, sizeof...(Us)>::call(
                    PIKA_MOVE(_args).template get<Is>(), PIKA_FORWARD(Us, vs)...)...);
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
# if defined(PIKA_HAVE_THREAD_DESCRIPTION)
            return pika::detail::get_function_annotation_itt<F>::call(_f);
# else
            static util::itt::string_handle sh("bound");
            return sh;
# endif
        }
#endif

    private:
        F _f;
        util::detail::member_pack_for<Ts...> _args;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    constexpr bound<std::decay_t<F>, util::detail::make_index_pack_t<sizeof...(Ts)>,
        ::pika::detail::decay_unwrap_t<Ts>...>
    bind(F&& f, Ts&&... vs)
    {
        using result_type = bound<std::decay_t<F>, util::detail::make_index_pack_t<sizeof...(Ts)>,
            ::pika::detail::decay_unwrap_t<Ts>...>;

        return result_type(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, vs)...);
    }
}    // namespace pika::util::detail

namespace pika::detail {
    template <typename F, typename... Ts>
    struct is_bind_expression<pika::util::detail::bound<F, Ts...>> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename... Ts>
    struct get_function_address<pika::util::detail::bound<F, Ts...>>
    {
        static constexpr std::size_t call(pika::util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<pika::util::detail::bound<F, Ts...>>
    {
        static constexpr char const* call(pika::util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

# if PIKA_HAVE_ITTNOTIFY != 0 && !defined(PIKA_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<pika::util::detail::bound<F, Ts...>>
    {
        static util::itt::string_handle call(pika::util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
# endif
#endif
}    // namespace pika::detail
