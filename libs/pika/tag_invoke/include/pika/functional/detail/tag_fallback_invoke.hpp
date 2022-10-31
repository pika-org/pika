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
        /// The `pika::functional::detail::tag_fallback_invoke` name defines a constexpr object
        /// that is invocable with one or more arguments. The first argument
        /// is a 'tag' (typically a CPO). It is only invocable if an overload
        /// of tag_fallback_invoke() that accepts the same arguments could be
        /// found via ADL.
        ///
        /// The evaluation of the expression
        /// `pika::functional::detail::tag_fallback_invoke(tag, args...)` is
        /// equivalent to evaluating the unqualified call to
        /// `tag_fallback_invoke(decay-copy(tag), PIKA_FORWARD(Args, args)...)`.
        ///
        /// `pika::functional::detail::tag_fallback_invoke` is implemented against P1895.
        ///
        /// Example:
        /// Defining a new customization point `foo`:
        /// ```
        /// namespace mylib {
        ///     inline constexpr
        ///         struct foo_fn final : pika::functional::detail::tag_fallback<foo_fn>
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
        ///     friend constexpr int tag_fallback_invoke(mylib::foo_fn, bar const& x)
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
        inline constexpr unspecified tag_fallback_invoke = unspecified;
    }    // namespace unspecified

    /// `pika::functional::is_tag_fallback_invocable<Tag, Args...>` is std::true_type if
    /// an overload of `tag_fallback_invoke(tag, args...)` can be found via ADL.
    template <typename Tag, typename... Args>
    struct is_tag_fallback_invocable;

    /// `pika::functional::is_tag_fallback_invocable_v<Tag, Args...>` evaluates to
    /// `pika::functional::is_tag_fallback_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_tag_fallback_invocable_v =
        is_tag_fallback_invocable<Tag, Args...>::value;

    /// `pika::functional::is_nothrow_tag_fallback_invocable<Tag, Args...>` is
    /// std::true_type if an overload of `tag_fallback_invoke(tag, args...)` can be
    /// found via ADL and is noexcept.
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_fallback_invocable;

    /// `pika::functional::is_tag_fallback_invocable_v<Tag, Args...>` evaluates to
    /// `pika::functional::is_tag_fallback_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_fallback_invocable_v =
        is_nothrow_tag_fallback_invocable<Tag, Args...>::value;

    /// `pika::functional::detail::tag_fallback_invoke_result<Tag, Args...>` is the trait
    /// returning the result type of the call pika::functional::detail::tag_fallback_invoke. This
    /// can be used in a SFINAE context.
    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result =
        invoke_result<decltype(tag_fallback_invoke), Tag, Args...>;

    /// `pika::functional::detail::tag_fallback_invoke_result_t<Tag, Args...>` evaluates to
    /// `pika::functional::detail::tag_fallback_invoke_result_t<Tag, Args...>::type`
    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result_t =
        typename tag_fallback_invoke_result<Tag, Args...>::type;

    /// `pika::functional::detail::tag_fallback<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    template <typename Tag>
    struct tag_fallback;

    /// `pika::functional::detail::tag_fallback_noexcept<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    /// where the implementation is required to be noexcept
    template <typename Tag>
    struct tag_fallback_noexcept;
}    // namespace pika::functional::detail
#else

#include <pika/config.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/functional/tag_invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace pika::functional::detail {
    namespace tag_fallback_invoke_t_ns {
        // poison pill
        void tag_fallback_invoke();

        struct tag_fallback_invoke_t
        {
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename Tag, typename... Ts>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Tag tag, Ts&&... ts) const
                noexcept(noexcept(tag_fallback_invoke(
                    std::declval<Tag>(), PIKA_FORWARD(Ts, ts)...)))
                    -> decltype(tag_fallback_invoke(
                        std::declval<Tag>(), PIKA_FORWARD(Ts, ts)...))
            {
                return tag_fallback_invoke(tag, PIKA_FORWARD(Ts, ts)...);
            }

            friend constexpr bool operator==(
                tag_fallback_invoke_t, tag_fallback_invoke_t)
            {
                return true;
            }

            friend constexpr bool operator!=(
                tag_fallback_invoke_t, tag_fallback_invoke_t)
            {
                return false;
            }
        };
    }    // namespace tag_fallback_invoke_t_ns

    namespace tag_fallback_invoke_ns {
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
        inline constexpr tag_fallback_invoke_t_ns::tag_fallback_invoke_t
            tag_fallback_invoke = {};
#else
        PIKA_DEVICE static tag_fallback_invoke_t_ns::tag_fallback_invoke_t const
            tag_fallback_invoke = {};
#endif
    }    // namespace tag_fallback_invoke_ns

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag, typename... Args>
    using is_tag_fallback_invocable = pika::detail::is_invocable<
        decltype(tag_fallback_invoke_ns::tag_fallback_invoke), Tag, Args...>;

    template <typename Tag, typename... Args>
    inline constexpr bool is_tag_fallback_invocable_v =
        is_tag_fallback_invocable<Tag, Args...>::value;

    template <typename Sig, bool Dispatchable>
    struct is_nothrow_tag_fallback_invocable_impl;

    template <typename Sig>
    struct is_nothrow_tag_fallback_invocable_impl<Sig, false> : std::false_type
    {
    };

    template <typename Tag, typename... Args>
    struct is_nothrow_tag_fallback_invocable_impl<
        decltype(tag_fallback_invoke_ns::tag_fallback_invoke)(Tag, Args...),
        true>
      : std::integral_constant<bool,
            noexcept(tag_fallback_invoke_ns::tag_fallback_invoke(
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
    struct is_nothrow_tag_fallback_invocable : std::true_type
    {
    };
#else
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_fallback_invocable
      : is_nothrow_tag_fallback_invocable_impl<
            decltype(tag_fallback_invoke_ns::tag_fallback_invoke)(Tag, Args...),
            is_tag_fallback_invocable_v<Tag, Args...>>
    {
    };
#endif

    template <typename Tag, typename... Args>
    inline constexpr bool is_nothrow_tag_fallback_invocable_v =
        is_nothrow_tag_fallback_invocable<Tag, Args...>::value;

    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result = pika::util::detail::invoke_result<
        decltype(tag_fallback_invoke_ns::tag_fallback_invoke), Tag, Args...>;

    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result_t =
        typename tag_fallback_invoke_result<Tag, Args...>::type;

    ///////////////////////////////////////////////////////////////////////////////
    namespace tag_base_ns {
        template <typename Tag, typename... Args>
        struct not_tag_fallback_noexcept_dispatchable;

        // poison pill
        void tag_fallback_invoke();

        ///////////////////////////////////////////////////////////////////////////
        /// Helper base class implementing the tag_invoke logic for CPOs that fall
        /// back to directly invoke its fallback.
        ///
        /// This base class is in many cases preferable to the plain tag base class.
        /// With the normal tag base class a default, unconstrained, default
        /// tag_invoke overload will take precedence over user-defined tag_invoke
        /// overloads that are not perfect matches. For example, with a default
        /// overload:
        ///
        /// template <typename T> auto tag_invoke(tag_t, T&& t) {...}
        ///
        /// and a user-defined overload in another namespace:
        ///
        /// auto tag_invoke(my_type t)
        ///
        /// the user-defined overload will only be considered when it is an exact
        /// match. This means const and reference qualifiers must match exactly, and
        /// conversions to a base class are not considered.
        ///
        /// With tag_fallback one can define the default implementation in terms of
        /// a tag_fallback_invoke overload instead of tag_invoke:
        ///
        /// template <typename T> auto tag_fallback_invoke(tag_t, T&& t) {...}
        ///
        /// With the same user-defined tag_invoke overload, the user-defined
        /// overload will now be used if it is a match even if it isn't an exact
        /// match.
        /// This is because tag_fallback will dispatch to tag_fallback_invoke only
        /// if there are no matching tag_invoke overloads.
        template <typename Tag>
        struct tag_fallback
        {
            // is tag-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable =
                    std::enable_if_t<is_tag_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const
                noexcept(is_nothrow_tag_invocable_v<Tag, Args...>)
                    -> tag_invoke_result_t<Tag, Args&&...>
            {
                return tag_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }

            // is not tag-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable =
                    std::enable_if_t<!is_tag_invocable_v<Tag, Args&&...>>>
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
        // helper base class implementing the tag_invoke logic for CPOs that fall
        // back to directly invoke its fallback. Either invocation has to be noexcept.
        template <typename Tag>
        struct tag_fallback_noexcept
        {
        private:
            // is nothrow tag-fallback dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args>
            PIKA_HOST_DEVICE constexpr auto tag_fallback_invoke_impl(
                std::false_type, Args&&... /*args*/) const noexcept
                -> not_tag_fallback_noexcept_dispatchable<Tag, Args...>
            {
                return not_tag_fallback_noexcept_dispatchable<Tag, Args...>{};
            }

            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            tag_fallback_invoke_impl(
                std::true_type, Args&&... args) const noexcept
                -> tag_fallback_invoke_result_t<Tag, Args&&...>
            {
                return tag_fallback_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }

        public:
            // is nothrow tag-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename Enable = std::enable_if_t<
                    is_nothrow_tag_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const noexcept
                -> tag_invoke_result_t<Tag, Args&&...>
            {
                return tag_invoke(static_cast<Tag const&>(*this),
                    PIKA_FORWARD(Args, args)...);
            }

            // is not nothrow tag-dispatchable
            PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
            template <typename... Args,
                typename IsFallbackDispatchable =
                    is_nothrow_tag_fallback_invocable<Tag, Args&&...>,
                typename Enable = std::enable_if_t<
                    !is_nothrow_tag_invocable_v<Tag, Args&&...>>>
            PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr auto
            operator()(Args&&... args) const noexcept
                -> decltype(tag_fallback_invoke_impl(
                    IsFallbackDispatchable{}, PIKA_FORWARD(Args, args)...))
            {
                return tag_fallback_invoke_impl(
                    IsFallbackDispatchable{}, PIKA_FORWARD(Args, args)...);
            }
        };
    }    // namespace tag_base_ns

    inline namespace tag_invoke_base_ns {
        template <typename Tag>
        using tag_fallback = tag_base_ns::tag_fallback<Tag>;

        template <typename Tag>
        using tag_fallback_noexcept = tag_base_ns::tag_fallback_noexcept<Tag>;
    }    // namespace tag_invoke_base_ns

    inline namespace tag_fallback_invoke_f_ns {
        using tag_fallback_invoke_ns::tag_fallback_invoke;
    }    // namespace tag_fallback_invoke_f_ns
}    // namespace pika::functional::detail

#endif
