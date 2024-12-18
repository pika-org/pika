//  Copyright (c) 2017-2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace pika::util::detail {
    ///////////////////////////////////////////////////////////////////////////
    // when `pm` is a pointer to member of a class `C` and
    // `is_base_of_v<C, remove_reference_t<T>>` is `true`;
    template <typename C, typename T,
        typename = typename std::enable_if<
            std::is_base_of<C, typename std::remove_reference<T>::type>::value>::type>
    static constexpr T&& mem_ptr_target(T&& v) noexcept
    {
        return std::forward<T>(v);
    }

    // when `pm` is a pointer to member of a class `C` and
    // `remove_cvref_t<T>` is a specialization of `reference_wrapper`;
    template <typename C, typename T>
    static constexpr T& mem_ptr_target(std::reference_wrapper<T> v) noexcept
    {
        return v.get();
    }

    // when `pm` is a pointer to member of a class `C` and `T` does not
    // satisfy the previous two items;
    template <typename C, typename T>
    static constexpr auto mem_ptr_target(T&& v) noexcept(
#if defined(PIKA_CUDA_VERSION)
        noexcept(*std::forward<T>(v))) -> decltype(*std::forward<T>(v))
#else
        noexcept(*std::forward<T>(v))) -> decltype(*std::forward<T>(v))
#endif
    {
#if defined(PIKA_CUDA_VERSION)
        return *std::forward<T>(v);
#else
        return *std::forward<T>(v);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename C>
    struct invoke_mem_obj
    {
        T C::*pm;

    public:
        constexpr invoke_mem_obj(T C::*pm) noexcept
          : pm(pm)
        {
        }

        template <typename T1>
        constexpr auto operator()(T1&& t1) const
            noexcept(noexcept(detail::mem_ptr_target<C>(std::forward<T1>(t1)).*
                pm)) -> decltype(detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm)
        {
            // This seems to trigger a bogus warning in GCC 11 with
            // optimizations enabled (possibly the same as this:
            // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
            // the warning locally.
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 110000
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Warray-bounds"
#endif
            return detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm;
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 110000
# pragma GCC diagnostic pop
#endif
        }
    };

    template <typename T, typename C>
    struct invoke_mem_fun
    {
        T C::*pm;

    public:
        constexpr invoke_mem_fun(T C::*pm) noexcept
          : pm(pm)
        {
        }

        template <typename T1, typename... Tn>
        constexpr auto operator()(T1&& t1, Tn&&... tn) const noexcept(
            noexcept((detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm)(std::forward<Tn>(
                tn)...))) -> decltype((detail::mem_ptr_target<C>(std::forward<T1>(t1)).*
                              pm)(std::forward<Tn>(tn)...))
        {
            // This seems to trigger a bogus warning in GCC 11 with
            // optimizations enabled (possibly the same as this:
            // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
            // the warning locally.
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 110000
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Warray-bounds"
#endif
            return (detail::mem_ptr_target<C>(std::forward<T1>(t1)).*pm)(std::forward<Tn>(tn)...);
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 110000
# pragma GCC diagnostic pop
#endif
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F,
        typename FD = typename std::remove_cv<typename std::remove_reference<F>::type>::type>
    struct dispatch_invoke
    {
        using type = F&&;
    };

    template <typename F, typename T, typename C>
    struct dispatch_invoke<F, T C::*>
    {
        using type = typename std::conditional<std::is_function<T>::value, invoke_mem_fun<T, C>,
            invoke_mem_obj<T, C>>::type;
    };

    template <typename F>
    using invoke_impl = typename dispatch_invoke<F>::type;

#define PIKA_INVOKE(F, ...) (::pika::util::detail::invoke_impl<decltype((F))>(F)(__VA_ARGS__))
}    // namespace pika::util::detail
