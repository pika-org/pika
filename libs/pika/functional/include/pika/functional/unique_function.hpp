//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/detail/basic_function.hpp>
#include <pika/functional/traits/get_function_address.hpp>
#include <pika/functional/traits/get_function_annotation.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace pika::util::detail {
    template <typename Sig>
    class unique_function;

    template <typename R, typename... Ts>
    class unique_function<R(Ts...)> : public detail::basic_function<R(Ts...), false>
    {
        using base_type = detail::basic_function<R(Ts...), false>;

    public:
        using result_type = R;

        constexpr unique_function(std::nullptr_t = nullptr) noexcept {}

        unique_function(unique_function&&) noexcept = default;
        unique_function& operator=(unique_function&&) noexcept = default;

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = std::decay_t<F>,
            typename Enable1 = std::enable_if_t<!std::is_same_v<FD, unique_function>>,
            typename Enable2 = std::enable_if_t<std::is_invocable_r_v<R, FD&, Ts...>>>
        unique_function(F&& f)
        {
            assign(PIKA_FORWARD(F, f));
        }

        // the split SFINAE prevents MSVC from eagerly instantiating things
        template <typename F, typename FD = std::decay_t<F>,
            typename Enable1 = std::enable_if_t<!std::is_same_v<FD, unique_function>>,
            typename Enable2 = std::enable_if_t<std::is_invocable_r_v<R, FD&, Ts...>>>
        unique_function& operator=(F&& f)
        {
            assign(PIKA_FORWARD(F, f));
            return *this;
        }

        using base_type::operator();
        using base_type::assign;
        using base_type::empty;
        using base_type::reset;
        using base_type::target;
    };
}    // namespace pika::util::detail

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace pika::detail {
    template <typename Sig>
    struct get_function_address<util::detail::unique_function<Sig>>
    {
        static constexpr std::size_t call(util::detail::unique_function<Sig> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename Sig>
    struct get_function_annotation<util::detail::unique_function<Sig>>
    {
        static constexpr char const* call(util::detail::unique_function<Sig> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

# if PIKA_HAVE_ITTNOTIFY != 0 && !defined(PIKA_HAVE_APEX)
    template <typename Sig>
    struct get_function_annotation_itt<util::detail::unique_function<Sig>>
    {
        static util::itt::string_handle call(util::detail::unique_function<Sig> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
# endif
}    // namespace pika::detail
#endif
