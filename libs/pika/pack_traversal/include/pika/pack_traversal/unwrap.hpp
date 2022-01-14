//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/pack_traversal/detail/unwrap_impl.hpp>

#include <cstddef>
#include <utility>

namespace pika {
    /// A helper function for retrieving the actual result of
    /// any pika::future like type which is wrapped in an arbitrary way.
    ///
    /// Unwraps the given pack of arguments, so that any pika::future
    /// object is replaced by its future result type in the argument pack:
    /// - `pika::future<int>` -> `int`
    /// - `pika::future<std::vector<float>>` -> `std::vector<float>`
    /// - `std::vector<future<float>>` -> `std::vector<float>`
    ///
    /// The function is capable of unwrapping pika::future like objects
    /// that are wrapped inside any container or tuple like type,
    /// see pika::util::map_pack() for a detailed description about which
    /// surrounding types are supported.
    /// Non pika::future like types are permitted as arguments and
    /// passed through.
    ///
    ///   ```cpp
    ///   // Single arguments
    ///   int i1 = pika:unwrap(pika::make_ready_future(0));
    ///
    ///   // Multiple arguments
    ///   pika::tuple<int, int> i2 =
    ///       pika:unwrap(pika::make_ready_future(1),
    ///                  pika::make_ready_future(2));
    ///   ```
    ///
    /// \note    This function unwraps the given arguments until the first
    ///          traversed nested pika::future which corresponds to
    ///          an unwrapping depth of one.
    ///          See pika::unwrap_n() for a function which unwraps the
    ///          given arguments to a particular depth or
    ///          pika::unwrap_all() that unwraps all future like objects
    ///          recursively which are contained in the arguments.
    ///
    /// \param   args the arguments that are unwrapped which may contain any
    ///          arbitrary future or non future type.
    ///
    /// \returns Depending on the count of arguments this function returns
    ///          a pika::tuple containing the unwrapped arguments
    ///          if multiple arguments are given.
    ///          In case the function is called with a single argument,
    ///          the argument is unwrapped and returned.
    ///
    /// \throws  std::exception like objects in case any of the given wrapped
    ///          pika::future objects were resolved through an exception.
    ///          See pika::future::get() for details.
    ///
    template <typename... Args>
    auto unwrap(Args&&... args) -> decltype(
        util::detail::unwrap_depth_impl<1U>(PIKA_FORWARD(Args, args)...))
    {
        return util::detail::unwrap_depth_impl<1U>(PIKA_FORWARD(Args, args)...);
    }

    namespace functional {
        /// A helper function object for functionally invoking
        /// `pika::unwrap`. For more information please refer to its
        /// documentation.
        struct unwrap
        {
            /// \cond NOINTERNAL
            template <typename... Args>
            auto operator()(Args&&... args)
                -> decltype(pika::unwrap(PIKA_FORWARD(Args, args)...))
            {
                return pika::unwrap(PIKA_FORWARD(Args, args)...);
            }
            /// \endcond
        };
    }    // namespace functional

    /// An alterntive version of pika::unwrap(), which unwraps the given
    /// arguments to a certain depth of pika::future like objects.
    ///
    /// \tparam Depth The count of pika::future like objects which are
    ///               unwrapped maximally.
    ///
    /// See unwrap for a detailed description.
    ///
    template <std::size_t Depth, typename... Args>
    auto unwrap_n(Args&&... args) -> decltype(
        util::detail::unwrap_depth_impl<Depth>(PIKA_FORWARD(Args, args)...))
    {
        static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
        return util::detail::unwrap_depth_impl<Depth>(
            PIKA_FORWARD(Args, args)...);
    }

    namespace functional {
        /// A helper function object for functionally invoking
        /// `pika::unwrap_n`. For more information please refer to its
        /// documentation.
        template <std::size_t Depth>
        struct unwrap_n
        {
            /// \cond NOINTERNAL
            template <typename... Args>
            auto operator()(Args&&... args)
                -> decltype(pika::unwrap_n<Depth>(PIKA_FORWARD(Args, args)...))
            {
                return pika::unwrap_n<Depth>(PIKA_FORWARD(Args, args)...);
            }
            /// \endcond
        };
    }    // namespace functional

    /// An alterntive version of pika::unwrap(), which unwraps the given
    /// arguments recursively so that all contained pika::future like
    /// objects are replaced by their actual value.
    ///
    /// See pika::unwrap() for a detailed description.
    ///
    template <typename... Args>
    auto unwrap_all(Args&&... args) -> decltype(
        util::detail::unwrap_depth_impl<0U>(PIKA_FORWARD(Args, args)...))
    {
        return util::detail::unwrap_depth_impl<0U>(PIKA_FORWARD(Args, args)...);
    }

    namespace functional {
        /// A helper function object for functionally invoking
        /// `pika::unwrap_all`. For more information please refer to its
        /// documentation.
        struct unwrap_all
        {
            /// \cond NOINTERNAL
            template <typename... Args>
            auto operator()(Args&&... args)
                -> decltype(pika::unwrap_all(PIKA_FORWARD(Args, args)...))
            {
                return pika::unwrap_all(PIKA_FORWARD(Args, args)...);
            }
            /// \endcond
        };
    }    // namespace functional

    /// Returns a callable object which unwraps its arguments upon
    /// invocation using the pika::unwrap() function and then passes
    /// the result to the given callable object.
    ///
    ///   ```cpp
    ///   auto callable = pika::unwrapping([](int left, int right) {
    ///       return left + right;
    ///   });
    ///
    ///   int i1 = callable(pika::make_ready_future(1),
    ///                     pika::make_ready_future(2));
    ///   ```
    ///
    /// See pika::unwrap() for a detailed description.
    ///
    /// \param callable the callable object which which is called with
    ///        the result of the corresponding unwrap function.
    ///
    template <typename T>
    auto unwrapping(T&& callable)
        -> decltype(util::detail::functional_unwrap_depth_impl<1U>(
            PIKA_FORWARD(T, callable)))
    {
        return util::detail::functional_unwrap_depth_impl<1U>(
            PIKA_FORWARD(T, callable));
    }

    /// Returns a callable object which unwraps its arguments upon
    /// invocation using the pika::unwrap_n() function and then passes
    /// the result to the given callable object.
    ///
    /// See pika::unwrapping() for a detailed description.
    ///
    template <std::size_t Depth, typename T>
    auto unwrapping_n(T&& callable)
        -> decltype(util::detail::functional_unwrap_depth_impl<Depth>(
            PIKA_FORWARD(T, callable)))
    {
        static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
        return util::detail::functional_unwrap_depth_impl<Depth>(
            PIKA_FORWARD(T, callable));
    }

    /// Returns a callable object which unwraps its arguments upon
    /// invocation using the pika::unwrap_all() function and then passes
    /// the result to the given callable object.
    ///
    /// See pika::unwrapping() for a detailed description.
    ///
    template <typename T>
    auto unwrapping_all(T&& callable)
        -> decltype(util::detail::functional_unwrap_depth_impl<0U>(
            PIKA_FORWARD(T, callable)))
    {
        return util::detail::functional_unwrap_depth_impl<0U>(
            PIKA_FORWARD(T, callable));
    }

    namespace util {
        template <typename... Args>
        PIKA_DEPRECATED_V(0, 1, "Please use pika::unwrap instead.")
        auto unwrap(Args&&... args) -> decltype(
            detail::unwrap_depth_impl<1U>(PIKA_FORWARD(Args, args)...))
        {
            return detail::unwrap_depth_impl<1U>(PIKA_FORWARD(Args, args)...);
        }

        namespace functional {
            struct PIKA_DEPRECATED_V(
                0, 1, "Please use pika::functional::unwrap instead.") unwrap
            {
                template <typename... Args>
                auto operator()(Args&&... args)
                    -> decltype(pika::unwrap(PIKA_FORWARD(Args, args)...))
                {
                    return pika::unwrap(PIKA_FORWARD(Args, args)...);
                }
            };
        }    // namespace functional

        template <std::size_t Depth, typename... Args>
        PIKA_DEPRECATED_V(0, 1, "Please use pika::unwrap_n instead.")
        auto unwrap_n(Args&&... args) -> decltype(
            detail::unwrap_depth_impl<Depth>(PIKA_FORWARD(Args, args)...))
        {
            static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
            return detail::unwrap_depth_impl<Depth>(PIKA_FORWARD(Args, args)...);
        }

        namespace functional {
            template <std::size_t Depth>
            struct PIKA_DEPRECATED_V(
                0, 1, "Please use pika::functional::unwrap instead.") unwrap_n
            {
                template <typename... Args>
                auto operator()(Args&&... args) -> decltype(
                    pika::unwrap_n<Depth>(PIKA_FORWARD(Args, args)...))
                {
                    return pika::unwrap_n<Depth>(PIKA_FORWARD(Args, args)...);
                }
            };
        }    // namespace functional

        template <typename... Args>
        PIKA_DEPRECATED_V(0, 1, "Please use pika::unwrap_all instead.")
        auto unwrap_all(Args&&... args) -> decltype(
            detail::unwrap_depth_impl<0U>(PIKA_FORWARD(Args, args)...))
        {
            return detail::unwrap_depth_impl<0U>(PIKA_FORWARD(Args, args)...);
        }

        namespace functional {
            struct PIKA_DEPRECATED_V(
                0, 1, "Please use pika::functional::unwrap instead.") unwrap_all
            {
                template <typename... Args>
                auto operator()(Args&&... args)
                    -> decltype(pika::unwrap_all(PIKA_FORWARD(Args, args)...))
                {
                    return pika::unwrap_all(PIKA_FORWARD(Args, args)...);
                }
            };
        }    // namespace functional

        template <typename T>
        PIKA_DEPRECATED_V(0, 1, "Please use pika::unwrapping instead.")
        auto unwrapping(T&& callable) -> decltype(
            detail::functional_unwrap_depth_impl<1U>(PIKA_FORWARD(T, callable)))
        {
            return detail::functional_unwrap_depth_impl<1U>(
                PIKA_FORWARD(T, callable));
        }

        template <std::size_t Depth, typename T>
        PIKA_DEPRECATED_V(0, 1, "Please use pika::unwrapping_n instead.")
        auto unwrapping_n(T&& callable)
            -> decltype(detail::functional_unwrap_depth_impl<Depth>(
                PIKA_FORWARD(T, callable)))
        {
            static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
            return detail::functional_unwrap_depth_impl<Depth>(
                PIKA_FORWARD(T, callable));
        }

        template <typename T>
        PIKA_DEPRECATED_V(0, 1, "Please use pika::unwrapping_all instead.")
        auto unwrapping_all(T&& callable) -> decltype(
            detail::functional_unwrap_depth_impl<0U>(PIKA_FORWARD(T, callable)))
        {
            return detail::functional_unwrap_depth_impl<0U>(
                PIKA_FORWARD(T, callable));
        }
    }    // namespace util
}    // namespace pika
