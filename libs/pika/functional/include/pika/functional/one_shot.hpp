//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/traits/get_function_address.hpp>
#include <pika/functional/traits/get_function_annotation.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika::util::detail {
    template <typename F>
    class one_shot_wrapper    //-V690
    {
    public:
        template <typename F_, typename = std::enable_if_t<std::is_constructible_v<F, F_>>>
        constexpr explicit one_shot_wrapper(F_&& f)
          : _f(PIKA_FORWARD(F_, f))
#if defined(PIKA_DEBUG)
          , _called(false)
#endif
        {
        }

        constexpr one_shot_wrapper(one_shot_wrapper&& other)
          : _f(std::move(other._f))
#if defined(PIKA_DEBUG)
          , _called(other._called)
#endif
        {
#if defined(PIKA_DEBUG)
            other._called = true;
#endif
        }

        void check_call()
        {
#if defined(PIKA_DEBUG)
            PIKA_ASSERT(!_called);
            _called = true;
#endif
        }

        template <typename... Ts>
        constexpr PIKA_HOST_DEVICE std::invoke_result_t<F, Ts...> operator()(Ts&&... vs)
        {
            check_call();

            return PIKA_INVOKE(std::move(_f), PIKA_FORWARD(Ts, vs)...);
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

    public:    // exposition-only
        F _f;
#if defined(PIKA_DEBUG)
        bool _called;
#endif
    };

    template <typename F>
    constexpr one_shot_wrapper<std::decay_t<F>> one_shot(F&& f)
    {
        using result_type = one_shot_wrapper<std::decay_t<F>>;

        return result_type(PIKA_FORWARD(F, f));
    }
}    // namespace pika::util::detail

namespace pika::detail {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    template <typename F>
    struct get_function_address<util::detail::one_shot_wrapper<F>>
    {
        static constexpr std::size_t call(util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct get_function_annotation<util::detail::one_shot_wrapper<F>>
    {
        static constexpr char const* call(util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
#endif
}    // namespace pika::detail
