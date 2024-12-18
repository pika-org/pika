//  Copyright (c) 2017-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
# include <pika/functional/detail/invoke.hpp>
# include <pika/functional/traits/get_function_address.hpp>
# include <pika/functional/traits/get_function_annotation.hpp>
# include <pika/threading_base/scoped_annotation.hpp>
# include <pika/threading_base/thread_description.hpp>
# include <pika/threading_base/thread_helpers.hpp>
# include <pika/type_support/decay.hpp>

# if defined(PIKA_HAVE_APEX)
#  include <pika/threading_base/external_timer.hpp>
# endif
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace pika {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename F>
        struct annotated_function
        {
            using fun_type = detail::decay_unwrap_t<F>;

            annotated_function() noexcept
              : name_(nullptr)
            {
            }

            annotated_function(F const& f, char const* name)
              : f_(f)
              , name_(name)
            {
            }

            annotated_function(F&& f, char const* name)
              : f_(std::move(f))
              , name_(name)
            {
            }

            template <typename... Ts>
            std::invoke_result_t<fun_type, Ts...> operator()(Ts&&... ts)
            {
                scoped_annotation annotate(get_function_annotation());
                return PIKA_INVOKE(f_, std::forward<Ts>(ts)...);
            }

            ///////////////////////////////////////////////////////////////////
            /// \brief Returns the function address
            ///
            /// This function returns the passed function address.
            /// \param none
            constexpr std::size_t get_function_address() const
            {
                return pika::detail::get_function_address<fun_type>::call(f_);
            }

            ///////////////////////////////////////////////////////////////////
            /// \brief Returns the function annotation
            ///
            /// This function returns the function annotation, if it has a name
            /// name is returned, name is returned; if name is empty the typeid
            /// is returned
            ///
            /// \param none
            constexpr char const* get_function_annotation() const noexcept
            {
                return name_ ? name_ : typeid(f_).name();
            }

            constexpr fun_type const& get_bound_function() const noexcept { return f_; }

        private:
            fun_type f_;
            char const* name_;
        };
    }    // namespace detail

    template <typename F>
    detail::annotated_function<std::decay_t<F>>
    annotated_function(F&& f, char const* name = nullptr)
    {
        using result_type = detail::annotated_function<std::decay_t<F>>;

        return result_type(std::forward<F>(f), name);
    }

    template <typename F>
    detail::annotated_function<std::decay_t<F>> annotated_function(F&& f, std::string name)
    {
        using result_type = detail::annotated_function<std::decay_t<F>>;

        // Store string in a set to ensure it lives for the entire duration of
        // the task.
        char const* name_c_str = pika::detail::store_function_annotation(std::move(name));
        return result_type(std::forward<F>(f), name_c_str);
    }

#else
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns a function annotated with the given annotation.
    ///
    /// Annotating includes setting the thread description per thread id.
    ///
    /// \param function
    template <typename F>
    constexpr F&& annotated_function(F&& f, char const* = nullptr) noexcept
    {
        return std::forward<F>(f);
    }

    template <typename F>
    constexpr F&& annotated_function(F&& f, std::string const&) noexcept
    {
        return std::forward<F>(f);
    }
#endif
}    // namespace pika

namespace pika::detail {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    template <typename F>
    struct get_function_address<pika::detail::annotated_function<F>>
    {
        static constexpr std::size_t call(pika::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename F>
    struct get_function_annotation<pika::detail::annotated_function<F>>
    {
        static constexpr char const* call(pika::detail::annotated_function<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };
#endif
}    // namespace pika::detail
