//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace pika { namespace functional {
    inline namespace unspecified {
        /// The `pika::functional::tag_invoke` name defines a constexpr object
        /// that is invocable with one or more arguments. The first argument
        /// is a 'tag' (typically a CPO). It is only invocable if an overload
        /// of tag_invoke() that accepts the same arguments could be found via
        /// ADL.
        ///
        /// The evaluation of the expression `pika::tag_invoke(tag, args...)` is
        /// equivalent to evaluating the unqualified call to
        /// `tag_invoke(decay-copy(tag), PIKA_FORWARD(Args, args)...)`.
        ///
        /// `pika::functional::tag_invoke` is implemented against P1895.
        ///
        /// Example:
        /// Defining a new customization point `foo`:
        /// ```
        /// namespace mylib {
        ///     inline constexpr
        ///         struct foo_fn final : pika::functional::tag<foo_fn>
        ///         {
        ///         } foo{};
        /// }
        /// ```
        ///
        /// Defining an object `bar` which customizes `foo`:
        /// ```
        /// struct bar
        /// {
        ///     int x = 42;
        ///
        ///     friend constexpr int tag_invoke(mylib::foo_fn, bar const& x)
        ///     {
        ///         return b.x;
        ///     }
        /// };
        /// ```
        ///
        /// Using the customization point:
        /// ```
        /// static_assert(42 == mylib::foo(bar{}), "The answer is 42");
        /// ```
        inline constexpr unspecified tag_invoke = unspecified;
    }    // namespace unspecified

    /// `pika::functional::is_tag_invocable<Tag, Args...>` is std::true_type if
    /// an overload of `tag_invoke(tag, args...)` can be found via ADL.
    template <typename Tag, typename... Args>
    struct is_tag_invocable;

    /// `pika::functional::is_tag_invocable_v<Tag, Args...>` evaluates to
    /// `pika::functional::is_tag_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_tag_invocable_v = is_tag_invocable<Tag, Args...>::value;

    /// `pika::functional::is_nothrow_tag_invocable<Tag, Args...>` is
    /// std::true_type if an overload of `tag_invoke(tag, args...)` can be
    /// found via ADL and is noexcept.
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_invocable;

    /// `pika::functional::is_tag_invocable_v<Tag, Args...>` evaluates to
    /// `pika::functional::is_tag_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_invocable_v =
        is_nothrow_tag_invocable<Tag, Args...>::value;

    /// `pika::functional::tag_invoke_result<Tag, Args...>` is the trait
    /// returning the result type of the call pika::functional::tag_invoke. This
    /// can be used in a SFINAE context.
    template <typename Tag, typename... Args>
    using tag_invoke_result = invoke_result<decltype(tag_invoke), Tag, Args...>;

    /// `pika::functional::tag_invoke_result_t<Tag, Args...>` evaluates to
    /// `pika::functional::tag_invoke_result_t<Tag, Args...>::type`
    template <typename Tag, typename... Args>
    using tag_invoke_result_t = typename tag_invoke_result<Tag, Args...>::type;

    /// `pika::functional::tag<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    template <typename Tag>
    struct tag;

    /// `pika::functional::tag_noexcept<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    /// The implementation has to be noexcept
    template <typename Tag>
    struct tag_noexcept;
}}    // namespace pika::functional
#else

#include <pika/config.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace pika { namespace functional {
    template <auto& Tag>
    using tag_t = std::decay_t<decltype(Tag)>;

    namespace tag_invoke_t_ns {
        // poison pill
        void tag_invoke();

        struct tag_invoke_t
        {
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename Tag, typename... Ts>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Tag tag, Ts&&... ts) const noexcept(noexcept(
                tag_invoke(std::declval<Tag>(), PIKA_FORWARD(Ts, ts)...)))
                -> decltype(tag_invoke(
                    std::declval<Tag>(), PIKA_FORWARD(Ts, ts)...))
            {
                return tag_invoke(tag, PIKA_FORWARD(Ts, ts)...);
            }

            friend constexpr bool operator==(tag_invoke_t, tag_invoke_t)
            {
                return true;
            }

            friend constexpr bool operator!=(tag_invoke_t, tag_invoke_t)
            {
                return false;
            }
        };
    }    // namespace tag_invoke_t_ns

    namespace tag_invoke_ns {
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
        inline constexpr tag_invoke_t_ns::tag_invoke_t tag_invoke = {};
#else
        PIKA_DEVICE static tag_invoke_t_ns::tag_invoke_t const tag_invoke = {};
#endif
    }    // namespace tag_invoke_ns

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag, typename... Args>
    using is_tag_invocable =
        pika::detail::is_invocable<decltype(tag_invoke_ns::tag_invoke), Tag,
            Args...>;

    template <typename Tag, typename... Args>
    inline constexpr bool is_tag_invocable_v =
        is_tag_invocable<Tag, Args...>::value;

    namespace detail {
        template <typename Sig, bool Dispatchable>
        struct is_nothrow_tag_invocable_impl;

        template <typename Sig>
        struct is_nothrow_tag_invocable_impl<Sig, false> : std::false_type
        {
        };

        template <typename Tag, typename... Args>
        struct is_nothrow_tag_invocable_impl<
            decltype(tag_invoke_ns::tag_invoke)(Tag, Args...), true>
          : std::integral_constant<bool,
                noexcept(tag_invoke_ns::tag_invoke(
                    std::declval<Tag>(), std::declval<Args>()...))>
        {
        };
    }    // namespace detail

    // CUDA versions less than 11.2 have a template instantiation bug which
    // leaves out certain template arguments and leads to us not being able to
    // correctly check this condition. We default to the more relaxed
    // noexcept(true) to not falsely exclude correct overloads. However, this
    // may lead to noexcept(false) overloads falsely being candidates.
#if defined(__NVCC__) && defined(PIKA_CUDA_VERSION) &&                         \
    (PIKA_CUDA_VERSION < 1102)
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_invocable : is_tag_invocable<Tag, Args...>
    {
    };
#else
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_invocable
      : detail::is_nothrow_tag_invocable_impl<
            decltype(tag_invoke_ns::tag_invoke)(Tag, Args...),
            is_tag_invocable_v<Tag, Args...>>
    {
    };
#endif

    template <typename Tag, typename... Args>
    inline constexpr bool is_nothrow_tag_invocable_v =
        is_nothrow_tag_invocable<Tag, Args...>::value;

    template <typename Tag, typename... Args>
    using tag_invoke_result =
        pika::util::detail::invoke_result<decltype(tag_invoke_ns::tag_invoke),
            Tag, Args...>;

    template <typename Tag, typename... Args>
    using tag_invoke_result_t = typename tag_invoke_result<Tag, Args...>::type;

    ///////////////////////////////////////////////////////////////////////////////
    namespace tag_base_ns {
        // poison pill
        void tag_invoke();

        ///////////////////////////////////////////////////////////////////////////
        // helper base class implementing the tag_invoke logic for CPOs
        template <typename Tag>
        struct tag
        {
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const
                noexcept(is_nothrow_tag_invocable_v<Tag, Args...>)
                    -> tag_invoke_result_t<Tag, Args...>
            {
                return tag_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }
        };

        template <typename Tag>
        struct tag_noexcept
        {
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable =
                    std::enable_if_t<is_nothrow_tag_invocable_v<Tag, Args...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const noexcept
                -> tag_invoke_result_t<Tag, decltype(args)...>
            {
                return tag_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }
        };
    }    // namespace tag_base_ns

    inline namespace tag_invoke_base_ns {
        template <typename Tag>
        using tag = tag_base_ns::tag<Tag>;

        template <typename Tag>
        using tag_noexcept = tag_base_ns::tag_noexcept<Tag>;
    }    // namespace tag_invoke_base_ns

    inline namespace tag_invoke_f_ns {
        using tag_invoke_ns::tag_invoke;
    }    // namespace tag_invoke_f_ns
}}       // namespace pika::functional

#endif
