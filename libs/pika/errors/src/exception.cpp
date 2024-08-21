//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(PIKA_HAVE_MODULE)
module;
#endif

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/logging.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/errors/error.hpp>
#include <pika/errors/error_code.hpp>
#include <pika/errors/exception.hpp>
#include <pika/errors/exception_info.hpp>

#if defined(PIKA_WINDOWS)
# include <process.h>
#elif defined(PIKA_HAVE_UNISTD_H)
# include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.errors;
#endif

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    /// Construct a pika::exception from a \a pika::error.
    ///
    /// \param e    The parameter \p e holds the pika::error code the new
    ///             exception should encapsulate.
    exception::exception(error e)
      : std::system_error(make_error_code(e, throwmode::plain))
    {
        PIKA_ASSERT((e >= pika::error::success && e < pika::error::last_error) ||
            (detail::error_code_has_system_error(static_cast<int>(e))));
        if (e != pika::error::success) { PIKA_LOG(err, "created exception: {}", this->what()); }
    }

    /// Construct a pika::exception from a std#system_error.
    exception::exception(std::system_error const& e)
      : std::system_error(e)
    {
        PIKA_LOG(err, "created exception: {}", this->what());
    }

    /// Construct a pika::exception from a std#system#error_code.
    exception::exception(std::error_code const& e)
      : std::system_error(e)
    {
        PIKA_LOG(err, "created exception: {}", this->what());
    }

    /// Construct a pika::exception from a \a pika::error and an error message.
    ///
    /// \param e      The parameter \p e holds the pika::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               pika::error_code belongs to the error category
    ///               \a pika_category (if mode is \a plain, this is the
    ///               default) or to the category \a pika_category_rethrow
    ///               (if mode is \a rethrow).
    exception::exception(error e, char const* msg, throwmode mode)
      : std::system_error(detail::make_system_error_code(e, mode), msg)
    {
        PIKA_ASSERT((e >= pika::error::success && e < pika::error::last_error) ||
            (detail::error_code_has_system_error(static_cast<int>(e))));
        if (e != pika::error::success) { PIKA_LOG(err, "created exception: {}", this->what()); }
    }

    /// Construct a pika::exception from a \a pika::error and an error message.
    ///
    /// \param e      The parameter \p e holds the pika::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               pika::error_code belongs to the error category
    ///               \a pika_category (if mode is \a plain, this is the
    ///               default) or to the category \a pika_category_rethrow
    ///               (if mode is \a rethrow).
    exception::exception(error e, std::string const& msg, throwmode mode)
      : std::system_error(detail::make_system_error_code(e, mode), msg)
    {
        PIKA_ASSERT((e >= pika::error::success && e < pika::error::last_error) ||
            (detail::error_code_has_system_error(static_cast<int>(e))));
        if (e != pika::error::success) { PIKA_LOG(err, "created exception: {}", this->what()); }
    }

    /// Destruct a pika::exception
    ///
    /// \throws nothing
    exception::~exception() noexcept {}

    /// The function \a get_error() returns the pika::error code stored
    /// in the referenced instance of a pika::exception. It returns
    /// the pika::error code this exception instance was constructed
    /// from.
    ///
    /// \throws nothing
    error exception::get_error() const noexcept
    {
        return static_cast<error>(this->std::system_error::code().value());
    }

    /// The function \a get_error_code() returns a pika::error_code which
    /// represents the same error condition as this pika::exception instance.
    ///
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               pika::error_code belongs to the error category
    ///               \a pika_category (if mode is \a plain, this is the
    ///               default) or to the category \a pika_category_rethrow
    ///               (if mode is \a rethrow).
    error_code exception::get_error_code(throwmode mode) const noexcept
    {
        (void) mode;
        return error_code(this->std::system_error::code().value(), *this);
    }

    namespace detail {
        static custom_exception_info_handler_type custom_exception_info_handler;

        void set_custom_exception_info_handler(custom_exception_info_handler_type f)
        {
            custom_exception_info_handler = f;
        }

        static pre_exception_handler_type pre_exception_handler;

        void set_pre_exception_handler(pre_exception_handler_type f) { pre_exception_handler = f; }
    }    // namespace detail
}    // namespace pika

namespace pika::detail {
    template <typename Exception>
    PIKA_EXPORT std::exception_ptr construct_lightweight_exception(
        Exception const& e, std::string const& func, std::string const& file, long line)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try
        {
            throw_with_info(e,
                std::move(pika::exception_info().set(pika::detail::throw_function(func),
                    pika::detail::throw_file(file), pika::detail::throw_line(line))));
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        PIKA_ASSERT(false);    // -V779
        return std::exception_ptr();
    }

    template <typename Exception>
    PIKA_EXPORT std::exception_ptr construct_lightweight_exception(Exception const& e)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try
        {
            pika::throw_with_info(e);
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        PIKA_ASSERT(false);    // -V779
        return std::exception_ptr();
    }

    template PIKA_EXPORT std::exception_ptr construct_lightweight_exception(
        pika::thread_interrupted const&);

    template <typename Exception>
    PIKA_EXPORT std::exception_ptr construct_custom_exception(Exception const& e,
        std::string const& func, std::string const& file, long line, std::string const& auxinfo)
    {
        if (!custom_exception_info_handler)
        {
            return construct_lightweight_exception(e, func, file, line);
        }

        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with information provided by the hook
        try
        {
            throw_with_info(e, custom_exception_info_handler(func, file, line, auxinfo));
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        PIKA_ASSERT(false);    // -V779
        return std::exception_ptr();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    inline bool is_of_lightweight_pika_category(Exception const&)
    {
        return false;
    }

    inline bool is_of_lightweight_pika_category(pika::exception const& e)
    {
        return e.get_error_code().category() == get_lightweight_pika_category();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::exception_ptr access_exception(error_code const& e) { return e.exception_; }

    template <typename Exception>
    PIKA_EXPORT std::exception_ptr get_exception(Exception const& e, std::string const& func,
        std::string const& file, long line, std::string const& auxinfo)
    {
        if (is_of_lightweight_pika_category(e))
        {
            return construct_lightweight_exception(e, func, file, line);
        }

        return construct_custom_exception(e, func, file, line, auxinfo);
    }

    template <typename Exception>
    PIKA_EXPORT void
    throw_exception(Exception const& e, std::string const& func, std::string const& file, long line)
    {
        if (pre_exception_handler) { pre_exception_handler(); }

        std::rethrow_exception(get_exception(e, func, file, line));
    }

    ///////////////////////////////////////////////////////////////////////////
    template PIKA_EXPORT std::exception_ptr get_exception(
        pika::exception const&, std::string const&, std::string const&, long, std::string const&);

    template PIKA_EXPORT std::exception_ptr get_exception(
        std::system_error const&, std::string const&, std::string const&, long, std::string const&);

    template PIKA_EXPORT std::exception_ptr get_exception(
        std::exception const&, std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(pika::detail::std_exception const&,
        std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(std::bad_exception const&,
        std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(pika::detail::bad_exception const&,
        std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(
        std::bad_typeid const&, std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(pika::detail::bad_typeid const&,
        std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(
        std::bad_cast const&, std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(pika::detail::bad_cast const&,
        std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(
        std::bad_alloc const&, std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(pika::detail::bad_alloc const&,
        std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(
        std::logic_error const&, std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(std::runtime_error const&,
        std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(
        std::out_of_range const&, std::string const&, std::string const&, long, std::string const&);
    template PIKA_EXPORT std::exception_ptr get_exception(std::invalid_argument const&,
        std::string const&, std::string const&, long, std::string const&);

    ///////////////////////////////////////////////////////////////////////////
    template PIKA_EXPORT void throw_exception(
        pika::exception const&, std::string const&, std::string const&, long);

    template PIKA_EXPORT void throw_exception(
        std::system_error const&, std::string const&, std::string const&, long);

    template PIKA_EXPORT void throw_exception(
        std::exception const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        pika::detail::std_exception const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::bad_exception const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        pika::detail::bad_exception const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::bad_typeid const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        pika::detail::bad_typeid const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::bad_cast const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        pika::detail::bad_cast const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::bad_alloc const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        pika::detail::bad_alloc const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::logic_error const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::runtime_error const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::out_of_range const&, std::string const&, std::string const&, long);
    template PIKA_EXPORT void throw_exception(
        std::invalid_argument const&, std::string const&, std::string const&, long);
}    // namespace pika::detail

///////////////////////////////////////////////////////////////////////////////
namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    /// Return the error message.
    std::string get_error_what(pika::exception_info const& xi)
    {
        // Try a cast to std::exception - this should handle boost.system
        // error codes in addition to the standard library exceptions.
        std::exception const* se = dynamic_cast<std::exception const*>(&xi);
        return se ? se->what() : std::string("<unknown>");
    }

    ///////////////////////////////////////////////////////////////////////////
    error get_error(pika::exception const& e) { return static_cast<pika::error>(e.get_error()); }

    error get_error(pika::error_code const& e) { return static_cast<pika::error>(e.value()); }

    error get_error(std::exception_ptr const& e)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (pika::thread_interrupted const&)
        {
            return pika::error::thread_cancelled;
        }
        catch (pika::exception const& he)
        {
            return he.get_error();
        }
        catch (std::system_error const& e)
        {
            int code = e.code().value();
            if (code < static_cast<int>(pika::error::success) ||
                code >= static_cast<int>(pika::error::last_error))
                code |= static_cast<int>(pika::error::system_error_flag);
            return static_cast<pika::error>(code);
        }
        catch (...)
        {
            return pika::error::unknown_error;
        }
    }

    /// Return the function name from which the exception was thrown.
    std::string get_error_function_name(pika::exception_info const& xi)
    {
        std::string const* function = xi.get<pika::detail::throw_function>();
        if (function) return *function;

        return std::string();
    }

    /// Return the (source code) file name of the function from which the
    /// exception was thrown.
    std::string get_error_file_name(pika::exception_info const& xi)
    {
        std::string const* file = xi.get<pika::detail::throw_file>();
        if (file) return *file;

        return "<unknown>";
    }

    /// Return the line number in the (source code) file of the function from
    /// which the exception was thrown.
    long get_error_line_number(pika::exception_info const& xi)
    {
        long const* line = xi.get<pika::detail::throw_line>();
        if (line) return *line;
        return -1;
    }

}    // namespace pika
