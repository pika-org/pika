//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2020-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace pika::functional::detail {
    inline namespace unspecified {
        /// The `pika::functional::detail::tag_override_invoke` name defines a constexpr object
        /// that is invocable with one or more arguments. The first argument
        /// is a 'tag' (typically a CPO). It is only invocable if an overload
        /// of tag_override_invoke() that accepts the same arguments could be
        /// found via ADL.
        ///
        /// The evaluation of the expression
        /// `pika::functional::detail::tag_override_invoke(tag, args...)` is
        /// equivalent to evaluating the unqualified call to
        /// `tag_override_invoke(decay-copy(tag), PIKA_FORWARD(Args, args)...)`.
        ///
        /// `pika::functional::detail::tag_override_invoke` is implemented against P1895.
        ///
        /// Example:
        /// Defining a new customization point `foo`:
        /// ```
        /// namespace mylib {
        ///     inline constexpr
        ///         struct foo_fn final : pika::functional::detail::tag_override<foo_fn>
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
        ///     friend constexpr int tag_override_invoke(mylib::foo_fn, bar const& x)
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
        inline constexpr unspecified tag_override_invoke = unspecified;
    }    // namespace unspecified

    /// `pika::functional::is_tag_override_invocable<Tag, Args...>` is std::true_type if
    /// an overload of `tag_override_invoke(tag, args...)` can be found via ADL.
    template <typename Tag, typename... Args>
    struct is_tag_override_invocable;

    /// `pika::functional::is_tag_override_invocable_v<Tag, Args...>` evaluates to
    /// `pika::functional::is_tag_override_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_tag_override_invocable_v =
        is_tag_override_invocable<Tag, Args...>::value;

    /// `pika::functional::is_nothrow_tag_override_invocable<Tag, Args...>` is
    /// std::true_type if an overload of `tag_override_invoke(tag, args...)` can be
    /// found via ADL and is noexcept.
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_override_invocable;

    /// `pika::functional::is_tag_override_invocable_v<Tag, Args...>` evaluates to
    /// `pika::functional::is_tag_override_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_override_invocable_v =
        is_nothrow_tag_override_invocable<Tag, Args...>::value;

    /// `pika::functional::detail::tag_override_invoke_result<Tag, Args...>` is the trait
    /// returning the result type of the call pika::functional::detail::tag_override_invoke. This
    /// can be used in a SFINAE context.
    template <typename Tag, typename... Args>
    using tag_override_invoke_result =
        invoke_result<decltype(tag_override_invoke), Tag, Args...>;

    /// `pika::functional::detail::tag_override_invoke_result_t<Tag, Args...>` evaluates to
    /// `pika::functional::detail::tag_override_invoke_result_t<Tag, Args...>::type`
    template <typename Tag, typename... Args>
    using tag_override_invoke_result_t =
        typename tag_override_invoke_result<Tag, Args...>::type;

    /// `pika::functional::detail::tag_override<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    template <typename Tag>
    struct tag_override;

    /// `pika::functional::detail::tag_override_noexcept<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    /// where the implementation is required to be noexcept
    template <typename Tag>
    struct tag_override_noexcept;
}    // namespace pika::functional::detail
#else

#include <pika/config.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace pika::functional::detail {
    namespace tag_override_invoke_t_ns {

        // poison pill
        void tag_override_invoke();

        struct tag_override_invoke_t
        {
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename Tag, typename... Ts>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Tag tag, Ts&&... ts) const
                noexcept(noexcept(tag_override_invoke(
                    std::declval<Tag>(), PIKA_FORWARD(Ts, ts)...)))
                    -> decltype(tag_override_invoke(
                        std::declval<Tag>(), PIKA_FORWARD(Ts, ts)...))
            {
                return tag_override_invoke(tag, PIKA_FORWARD(Ts, ts)...);
            }

            friend constexpr bool operator==(
                tag_override_invoke_t, tag_override_invoke_t)
            {
                return true;
            }

            friend constexpr bool operator!=(
                tag_override_invoke_t, tag_override_invoke_t)
            {
                return false;
            }
        };
    }    // namespace tag_override_invoke_t_ns

    namespace tag_override_invoke_ns {
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
        inline constexpr tag_override_invoke_t_ns::tag_override_invoke_t
            tag_override_invoke = {};
#else
        PIKA_DEVICE static tag_override_invoke_t_ns::tag_override_invoke_t const
            tag_override_invoke = {};
#endif
    }    // namespace tag_override_invoke_ns

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag, typename... Args>
    using is_tag_override_invocable = pika::detail::is_invocable<
        decltype(tag_override_invoke_ns::tag_override_invoke), Tag, Args...>;

    template <typename Tag, typename... Args>
    inline constexpr bool is_tag_override_invocable_v =
        is_tag_override_invocable<Tag, Args...>::value;

    template <typename Sig, bool Dispatchable>
    struct is_nothrow_tag_override_invocable_impl;

    template <typename Sig>
    struct is_nothrow_tag_override_invocable_impl<Sig, false> : std::false_type
    {
    };

    template <typename Tag, typename... Args>
    struct is_nothrow_tag_override_invocable_impl<
        decltype(tag_override_invoke_ns::tag_override_invoke)(Tag, Args...),
        true>
      : std::integral_constant<bool,
            noexcept(tag_override_invoke_ns::tag_override_invoke(
                std::declval<Tag>(), std::declval<Args>()...))>
    {
    };

    // CUDA versions less than 11.2 have a template instantiation bug which
    // leaves out certain template arguments and leads to us not being able to
    // correctly check this condition. We default to the more relaxed
    // noexcept(true) to not falsely exclude correct overloads. However, this
    // may lead to noexcept(false) overloads falsely being candidates.
#if defined(__NVCC__) && defined(PIKA_CUDA_VERSION) &&                         \
    (PIKA_CUDA_VERSION < 1102)
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_override_invocable : std::true_type
    {
    };
#else
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_override_invocable
      : is_nothrow_tag_override_invocable_impl<
            decltype(tag_override_invoke_ns::tag_override_invoke)(Tag, Args...),
            is_tag_override_invocable_v<Tag, Args...>>
    {
    };
#endif

    template <typename Tag, typename... Args>
    inline constexpr bool is_nothrow_tag_override_invocable_v =
        is_nothrow_tag_override_invocable<Tag, Args...>::value;

    template <typename Tag, typename... Args>
    using tag_override_invoke_result = pika::util::detail::invoke_result<
        decltype(tag_override_invoke_ns::tag_override_invoke), Tag, Args...>;

    template <typename Tag, typename... Args>
    using tag_override_invoke_result_t =
        typename tag_override_invoke_result<Tag, Args...>::type;

    namespace tag_base_ns {
        // poison pill
        void tag_override_invoke();

        ///////////////////////////////////////////////////////////////////////////
        /// Helper base class implementing the tag_invoke logic for CPOs that allow
        /// overriding user-defined tag_invoke overloads with tag_override_invoke,
        /// and that allow setting a fallback with tag_fallback_invoke.
        ///
        /// This helper class is otherwise identical to tag_fallback, but allows
        /// defining an implementation that will always take priority if it is
        /// feasible. This is useful for example in cases where a member function
        /// should always take priority over any free function tag_invoke overloads,
        /// when available, like this:
        ///
        /// template <typename T>
        /// auto tag_override_invoke(T&& t) -> decltype(t.foo()){ return t.foo(); }
        template <typename Tag>
        struct tag_priority
        {
            // Is tag-override-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable = std::enable_if_t<
                    is_tag_override_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const
                noexcept(is_nothrow_tag_override_invocable_v<Tag, Args...>)
                    -> tag_override_invoke_result_t<Tag, Args&&...>
            {
                return tag_override_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }

            // Is not tag-override-dispatchable, but tag-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable = std::enable_if_t<
                    !is_tag_override_invocable_v<Tag, Args&&...> &&
                    is_tag_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const
                noexcept(is_nothrow_tag_invocable_v<Tag, Args...>)
                    -> tag_invoke_result_t<Tag, Args&&...>
            {
                return tag_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }

            // Is not tag-override-dispatchable, not tag-dispatchable, but
            // tag-fallback-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable = std::enable_if_t<
                    !is_tag_override_invocable_v<Tag, Args&&...> &&
                    !is_tag_invocable_v<Tag, Args&&...> &&
                    is_tag_fallback_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const
                noexcept(is_nothrow_tag_fallback_invocable_v<Tag, Args...>)
                    -> tag_fallback_invoke_result_t<Tag, Args&&...>
            {
                return tag_fallback_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////////
        // Helper base class implementing the tag_invoke logic for noexcept CPOs
        // that allow overriding user-defined tag_invoke overloads with
        // tag_override_invoke, and that allow setting a fallback with
        // tag_fallback_invoke.
        template <typename Tag>
        struct tag_priority_noexcept
        {
            // Is nothrow tag-override-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable = std::enable_if_t<
                    is_nothrow_tag_override_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const noexcept
                -> tag_override_invoke_result_t<Tag, Args&&...>
            {
                return tag_override_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }

            // Is not nothrow tag-override-dispatchable, but nothrow
            // tag-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable = std::enable_if_t<
                    !is_nothrow_tag_override_invocable_v<Tag, Args&&...> &&
                    is_nothrow_tag_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const noexcept
                -> tag_invoke_result_t<Tag, Args&&...>
            {
                return tag_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }

            // Is not nothrow tag-override-dispatchable, not nothrow
            // tag-dispatchable, but nothrow tag-fallback-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable = std::enable_if_t<
                    !is_nothrow_tag_override_invocable_v<Tag, Args&&...> &&
                    !is_nothrow_tag_invocable_v<Tag, Args&&...> &&
                    is_nothrow_tag_fallback_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const noexcept
                -> tag_fallback_invoke_result_t<Tag, Args&&...>
            {
                return tag_fallback_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }
        };
    }    // namespace tag_base_ns

    inline namespace tag_invoke_base_ns {
        template <typename Tag>
        using tag_priority = tag_base_ns::tag_priority<Tag>;

        template <typename Tag>
        using tag_priority_noexcept = tag_base_ns::tag_priority_noexcept<Tag>;
    }    // namespace tag_invoke_base_ns

    inline namespace tag_override_invoke_f_ns {
        using tag_override_invoke_ns::tag_override_invoke;
    }    // namespace tag_override_invoke_f_ns
}    // namespace pika::functional::detail

#endif
