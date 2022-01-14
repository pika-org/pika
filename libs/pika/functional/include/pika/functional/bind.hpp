//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/datastructures/member_pack.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/functional/one_shot.hpp>
#include <pika/functional/traits/get_function_address.hpp>
#include <pika/functional/traits/get_function_annotation.hpp>
#include <pika/functional/traits/is_action.hpp>
#include <pika/functional/traits/is_bind_expression.hpp>
#include <pika/functional/traits/is_placeholder.hpp>
#include <pika/type_support/decay.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika { namespace util {
    namespace placeholders {
        using namespace std::placeholders;
    }    // namespace placeholders

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <std::size_t I>
        struct bind_eval_placeholder
        {
            template <typename T, typename... Us>
            static constexpr PIKA_HOST_DEVICE decltype(auto) call(
                T&& /*t*/, Us&&... vs)
            {
                return util::member_pack_for<Us&&...>(
                    std::piecewise_construct, PIKA_FORWARD(Us, vs)...)
                    .template get<I>();
            }
        };

        template <typename T, std::size_t NumUs, typename TD = std::decay_t<T>,
            typename Enable = void>
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
            std::enable_if_t<traits::is_placeholder_v<TD> != 0 &&
                (traits::is_placeholder_v<TD> <= NumUs)>>
          : bind_eval_placeholder<(std::size_t) traits::is_placeholder_v<TD> -
                1>
        {
        };

        template <typename T, std::size_t NumUs, typename TD>
        struct bind_eval<T, NumUs, TD,
            std::enable_if_t<traits::is_bind_expression_v<TD>>>
        {
            template <typename... Us>
            static constexpr PIKA_HOST_DEVICE util::invoke_result_t<T, Us...>
            call(T&& t, Us&&... vs)
            {
                return PIKA_INVOKE(PIKA_FORWARD(T, t), PIKA_FORWARD(Us, vs)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename... Us>
        struct invoke_bound_result;

        template <typename F, typename... Ts, typename... Us>
        struct invoke_bound_result<F, util::pack<Ts...>, Us...>
          : util::invoke_result<F,
                decltype(bind_eval<Ts, sizeof...(Us)>::call(
                    std::declval<Ts>(), std::declval<Us>()...))...>
        {
        };

        template <typename F, typename Ts, typename... Us>
        using invoke_bound_result_t =
            typename invoke_bound_result<F, Ts, Us...>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Is, typename... Ts>
        class bound;

        template <typename F, std::size_t... Is, typename... Ts>
        class bound<F, index_pack<Is...>, Ts...>
        {
        public:
            bound() = default;    // needed for serialization

            template <typename F_, typename... Ts_,
                typename =
                    std::enable_if_t<std::is_constructible<F, F_>::value>>
            constexpr explicit bound(F_&& f, Ts_&&... vs)
              : _f(PIKA_FORWARD(F_, f))
              , _args(std::piecewise_construct, PIKA_FORWARD(Ts_, vs)...)
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound(bound const&) = default;
            bound(bound&&) = default;
#else
            constexpr PIKA_HOST_DEVICE bound(bound const& other)
              : _f(other._f)
              , _args(other._args)
            {
            }

            constexpr PIKA_HOST_DEVICE bound(bound&& other)
              : _f(PIKA_MOVE(other._f))
              , _args(PIKA_MOVE(other._args))
            {
            }
#endif

            bound& operator=(bound const&) = delete;

            template <typename... Us>
            constexpr PIKA_HOST_DEVICE
                invoke_bound_result_t<F&, util::pack<Ts&...>, Us&&...>
                operator()(Us&&... vs) &
            {
                return PIKA_INVOKE(_f,
                    detail::bind_eval<Ts&, sizeof...(Us)>::call(
                        _args.template get<Is>(), PIKA_FORWARD(Us, vs)...)...);
            }

            template <typename... Us>
            constexpr PIKA_HOST_DEVICE invoke_bound_result_t<F const&,
                util::pack<Ts const&...>, Us&&...>
            operator()(Us&&... vs) const&
            {
                return PIKA_INVOKE(_f,
                    detail::bind_eval<Ts const&, sizeof...(Us)>::call(
                        _args.template get<Is>(), PIKA_FORWARD(Us, vs)...)...);
            }

            template <typename... Us>
            constexpr PIKA_HOST_DEVICE
                invoke_bound_result_t<F&&, util::pack<Ts&&...>, Us&&...>
                operator()(Us&&... vs) &&
            {
                return PIKA_INVOKE(PIKA_MOVE(_f),
                    detail::bind_eval<Ts, sizeof...(Us)>::call(
                        PIKA_MOVE(_args).template get<Is>(),
                        PIKA_FORWARD(Us, vs)...)...);
            }

            template <typename... Us>
            constexpr PIKA_HOST_DEVICE invoke_bound_result_t<F const&&,
                util::pack<Ts const&&...>, Us&&...>
            operator()(Us&&... vs) const&&
            {
                return PIKA_INVOKE(PIKA_MOVE(_f),
                    detail::bind_eval<Ts const, sizeof...(Us)>::call(
                        PIKA_MOVE(_args).template get<Is>(),
                        PIKA_FORWARD(Us, vs)...)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                // clang-format off
                ar & _f;
                ar & _args;
                // clang-format on
            }

            constexpr std::size_t get_function_address() const
            {
                return traits::get_function_address<F>::call(_f);
            }

            constexpr char const* get_function_annotation() const
            {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation<F>::call(_f);
#else
                return nullptr;
#endif
            }

#if PIKA_HAVE_ITTNOTIFY != 0 && !defined(PIKA_HAVE_APEX)
            util::itt::string_handle get_function_annotation_itt() const
            {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation_itt<F>::call(_f);
#else
                static util::itt::string_handle sh("bound");
                return sh;
#endif
            }
#endif

        private:
            F _f;
            util::member_pack_for<Ts...> _args;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts,
        typename Enable =
            std::enable_if_t<!traits::is_action_v<std::decay_t<F>>>>
    constexpr detail::bound<std::decay_t<F>,
        util::make_index_pack_t<sizeof...(Ts)>, util::decay_unwrap_t<Ts>...>
    bind(F&& f, Ts&&... vs)
    {
        using result_type = detail::bound<std::decay_t<F>,
            util::make_index_pack_t<sizeof...(Ts)>,
            util::decay_unwrap_t<Ts>...>;

        return result_type(PIKA_FORWARD(F, f), PIKA_FORWARD(Ts, vs)...);
    }
}}    // namespace pika::util

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct is_bind_expression<util::detail::bound<F, Ts...>> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename... Ts>
    struct get_function_address<util::detail::bound<F, Ts...>>
    {
        static constexpr std::size_t call(
            util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<util::detail::bound<F, Ts...>>
    {
        static constexpr char const* call(
            util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if PIKA_HAVE_ITTNOTIFY != 0 && !defined(PIKA_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<util::detail::bound<F, Ts...>>
    {
        static util::itt::string_handle call(
            util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
#endif
}}    // namespace pika::traits

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace serialization {

    // serialization of the bound object
    template <typename Archive, typename F, typename... Ts>
    void serialize(Archive& ar, ::pika::util::detail::bound<F, Ts...>& bound,
        unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }

    // serialization of placeholders is trivial, just provide empty functions
    template <typename Archive, int I>
    void serialize(Archive& /* ar */,
        std::integral_constant<int, I>& /*placeholder*/
        ,
        unsigned int const /*version*/ = 0)
    {
    }
}}    // namespace pika::serialization
