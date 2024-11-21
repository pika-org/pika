//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/functional/traits/get_function_address.hpp>
#include <pika/functional/traits/get_function_annotation.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>
#include <pika/type_support/unused.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>

namespace pika::detail {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
    struct thread_description
    {
    public:
        enum data_type
        {
            data_type_description = 0,
            data_type_address = 1
        };

    private:
        union data
        {
            char const* desc_;    //-V117
            std::size_t addr_;    //-V117
        };

        data_type type_;
        data data_;

        PIKA_EXPORT void init_from_alternative_name(char const* altname);

    public:
        thread_description() noexcept
          : type_(data_type_description)
        {
            data_.desc_ = "<unknown>";
        }

        thread_description(char const* desc) noexcept
          : type_(data_type_description)
        {
            data_.desc_ = desc ? desc : "<unknown>";
        }

        /* The priority of description is name, altname, address */
        template <typename F, typename = std::enable_if_t<!std::is_same_v<F, thread_description>>>
        explicit thread_description(F const& f, char const* altname = nullptr) noexcept
          : type_(data_type_description)
        {
            char const* name = pika::detail::get_function_annotation<F>::call(f);

            // If a name exists, use it, not the altname.
            if (name != nullptr)    // -V547
            {
                altname = name;
            }

# if defined(PIKA_HAVE_THREAD_DESCRIPTION_FULL)
            if (altname != nullptr) { data_.desc_ = altname; }
            else
            {
                type_ = data_type_address;
                data_.addr_ = pika::detail::get_function_address<F>::call(f);
            }
# else
            init_from_alternative_name(altname);
# endif
        }

        constexpr data_type kind() const noexcept { return type_; }

        char const* get_description() const noexcept
        {
            PIKA_ASSERT(type_ == data_type_description);
            return data_.desc_;
        }

        std::size_t get_address() const noexcept
        {
            PIKA_ASSERT(type_ == data_type_address);
            return data_.addr_;
        }

        explicit operator bool() const noexcept { return valid(); }

        bool valid() const noexcept
        {
            if (type_ == data_type_description) return nullptr != data_.desc_;

            PIKA_ASSERT(type_ == data_type_address);
            return 0 != data_.addr_;
        }
    };
#else
    ///////////////////////////////////////////////////////////////////////////
    struct thread_description
    {
    public:
        enum data_type
        {
            data_type_description = 0,
            data_type_address = 1
        };

    private:
        // expose for ABI compatibility reasons
        PIKA_EXPORT void init_from_alternative_name(char const* altname);

    public:
        thread_description() noexcept = default;

        constexpr thread_description(char const* /*desc*/) noexcept {}

        template <typename F,
            typename = typename std::enable_if_t<!std::is_same_v<F, thread_description>>>
        explicit constexpr thread_description(
            F const& /*f*/, char const* /*altname*/ = nullptr) noexcept
        {
        }

        constexpr data_type kind() const noexcept { return data_type_description; }

        constexpr char const* get_description() const noexcept { return "<unknown>"; }

        constexpr std::size_t get_address() const noexcept { return 0; }

        explicit constexpr operator bool() const noexcept { return valid(); }

        constexpr bool valid() const noexcept { return true; }
    };
#endif

    PIKA_EXPORT std::ostream& operator<<(std::ostream&, thread_description const&);
    PIKA_EXPORT std::string as_string(thread_description const& desc);
}    // namespace pika::detail

namespace pika::threads::detail {
    ///////////////////////////////////////////////////////////////////////////
    /// The function get_thread_description is part of the thread related API
    /// allows to query the description of one of the threads known to the
    /// thread-manager.
    ///
    /// \param id         [in] The thread id of the thread being queried.
    /// \param ec         [in,out] this represents the error status on exit,
    ///                   if this is pre-initialized to \a pika#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns          This function returns the description of the
    ///                   thread referenced by the \a id parameter. If the
    ///                   thread is not known to the thread-manager the return
    ///                   value will be the string "<unknown>".
    ///
    /// \note             As long as \a ec is not pre-initialized to
    ///                   \a pika#throws this function doesn't
    ///                   throw but returns the result code using the
    ///                   parameter \a ec. Otherwise it throws an instance
    ///                   of pika#exception.
    PIKA_EXPORT ::pika::detail::thread_description get_thread_description(
        thread_id_type const& id, error_code& ec = throws);
    PIKA_EXPORT ::pika::detail::thread_description set_thread_description(thread_id_type const& id,
        ::pika::detail::thread_description const& desc = ::pika::detail::thread_description(),
        error_code& ec = throws);

    PIKA_EXPORT ::pika::detail::thread_description get_thread_lco_description(
        thread_id_type const& id, error_code& ec = throws);
    PIKA_EXPORT ::pika::detail::thread_description set_thread_lco_description(
        thread_id_type const& id,
        ::pika::detail::thread_description const& desc = ::pika::detail::thread_description(),
        error_code& ec = throws);
}    // namespace pika::threads::detail

template <>
struct fmt::formatter<pika::detail::thread_description> : fmt::formatter<std::string>
{
    template <typename FormatContext>
    auto format(pika::detail::thread_description const& desc, FormatContext& ctx) const
    {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
        if (desc.kind() == pika::detail::thread_description::data_type_description)
            return fmt::formatter<std::string>::format(
                desc ? desc.get_description() : "<unknown>", ctx);

        return fmt::formatter<std::string>::format(fmt::format("{}", desc.get_address()), ctx);
#else
        PIKA_UNUSED(desc);
        return fmt::formatter<std::string>::format("<unknown>", ctx);
#endif
    }
};
