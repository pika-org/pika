//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file error_code.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/errors/error.hpp>
#include <pika/errors/exception_fwd.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    /// \cond NODETAIL
    namespace detail {
        PIKA_EXPORT std::exception_ptr access_exception(error_code const&);

        ///////////////////////////////////////////////////////////////////////
        struct command_line_error : std::logic_error
        {
            explicit command_line_error(char const* msg)
              : std::logic_error(msg)
            {
            }

            explicit command_line_error(std::string const& msg)
              : std::logic_error(msg)
            {
            }
        };
    }    // namespace detail
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns generic pika error category used for new errors.
    PIKA_EXPORT std::error_category const& get_pika_category();

    /// \brief Returns generic pika error category used for errors re-thrown
    ///        after the exception has been de-serialized.
    PIKA_EXPORT std::error_category const& get_pika_rethrow_category();

    /// \cond NOINTERNAL
    PIKA_EXPORT std::error_category const& get_lightweight_pika_category();

    PIKA_EXPORT std::error_category const& get_pika_category(
        throwmode mode);

    inline std::error_code make_system_error_code(
        error e, throwmode mode = plain)
    {
        return std::error_code(static_cast<int>(e), get_pika_category(mode));
    }

    ///////////////////////////////////////////////////////////////////////////
    inline std::error_condition make_error_condition(error e, throwmode mode)
    {
        return std::error_condition(
            static_cast<int>(e), get_pika_category(mode));
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief A pika::error_code represents an arbitrary error condition.
    ///
    /// The class pika::error_code describes an object used to hold error code
    /// values, such as those originating from the operating system or other
    /// low-level application program interfaces.
    ///
    /// \note Class pika::error_code is an adjunct to error reporting by
    /// exception
    ///
    class error_code : public std::error_code    //-V690
    {
    public:
        /// Construct an object of type error_code.
        ///
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               pika::error_code belongs to the error category
        ///               \a pika_category (if mode is \a plain, this is the
        ///               default) or to the category \a pika_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        explicit error_code(throwmode mode = plain)
          : std::error_code(make_system_error_code(success, mode))
        {
        }

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the pika::error code the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               pika::error_code belongs to the error category
        ///               \a pika_category (if mode is \a plain, this is the
        ///               default) or to the category \a pika_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        PIKA_EXPORT explicit error_code(error e, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the pika::error code the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               pika::error_code belongs to the error category
        ///               \a pika_category (if mode is \a plain, this is the
        ///               default) or to the category \a pika_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws nothing
        PIKA_EXPORT error_code(error e, char const* func, char const* file,
            long line, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the pika::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               pika::error_code belongs to the error category
        ///               \a pika_category (if mode is \a plain, this is the
        ///               default) or to the category \a pika_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        PIKA_EXPORT error_code(
            error e, char const* msg, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the pika::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               pika::error_code belongs to the error category
        ///               \a pika_category (if mode is \a plain, this is the
        ///               default) or to the category \a pika_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        PIKA_EXPORT error_code(error e, char const* msg, char const* func,
            char const* file, long line, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the pika::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               pika::error_code belongs to the error category
        ///               \a pika_category (if mode is \a plain, this is the
        ///               default) or to the category \a pika_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        PIKA_EXPORT error_code(
            error e, std::string const& msg, throwmode mode = plain);

        /// Construct an object of type error_code.
        ///
        /// \param e      The parameter \p e holds the pika::error code the new
        ///               exception should encapsulate.
        /// \param msg    The parameter \p msg holds the error message the new
        ///               exception should encapsulate.
        /// \param func   The name of the function where the error was raised.
        /// \param file   The file name of the code where the error was raised.
        /// \param line   The line number of the code line where the error was
        ///               raised.
        /// \param mode   The parameter \p mode specifies whether the constructed
        ///               pika::error_code belongs to the error category
        ///               \a pika_category (if mode is \a plain, this is the
        ///               default) or to the category \a pika_category_rethrow
        ///               (if mode is \a rethrow).
        ///
        /// \throws std#bad_alloc (if allocation of a copy of
        ///         the passed string fails).
        PIKA_EXPORT error_code(error e, std::string const& msg,
            char const* func, char const* file, long line,
            throwmode mode = plain);

        /// Return a reference to the error message stored in the pika::error_code.
        ///
        /// \throws nothing
        PIKA_EXPORT std::string get_message() const;

        /// \brief Clear this error_code object.
        /// The postconditions of invoking this method are
        /// * value() == pika::success and category() == pika::get_pika_category()
        void clear()
        {
            this->std::error_code::assign(success, get_pika_category());
            exception_ = std::exception_ptr();
        }

        /// Copy constructor for error_code
        ///
        /// \note This function maintains the error category of the left hand
        ///       side if the right hand side is a success code.
        PIKA_EXPORT error_code(error_code const& rhs);

        /// Assignment operator for error_code
        ///
        /// \note This function maintains the error category of the left hand
        ///       side if the right hand side is a success code.
        PIKA_EXPORT error_code& operator=(error_code const& rhs);

    private:
        friend std::exception_ptr detail::access_exception(error_code const&);
        friend class exception;
        friend error_code make_error_code(std::exception_ptr const&);

        PIKA_EXPORT error_code(int err, pika::exception const& e);
        PIKA_EXPORT explicit error_code(std::exception_ptr const& e);

        std::exception_ptr exception_;
    };

    /// @{
    /// \brief Returns a new error_code constructed from the given parameters.
    inline error_code make_error_code(error e, throwmode mode = plain)
    {
        return error_code(e, mode);
    }
    inline error_code make_error_code(error e, char const* func,
        char const* file, long line, throwmode mode = plain)
    {
        return error_code(e, func, file, line, mode);
    }

    /// \brief Returns error_code(e, msg, mode).
    inline error_code make_error_code(
        error e, char const* msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }
    inline error_code make_error_code(error e, char const* msg,
        char const* func, char const* file, long line, throwmode mode = plain)
    {
        return error_code(e, msg, func, file, line, mode);
    }

    /// \brief Returns error_code(e, msg, mode).
    inline error_code make_error_code(
        error e, std::string const& msg, throwmode mode = plain)
    {
        return error_code(e, msg, mode);
    }
    inline error_code make_error_code(error e, std::string const& msg,
        char const* func, char const* file, long line, throwmode mode = plain)
    {
        return error_code(e, msg, func, file, line, mode);
    }
    inline error_code make_error_code(std::exception_ptr const& e)
    {
        return error_code(e);
    }
    ///@}

    /// \brief Returns error_code(pika::success, "success", mode).
    inline error_code make_success_code(throwmode mode = plain)
    {
        return error_code(mode);
    }
}    // namespace pika

#include <pika/errors/throw_exception.hpp>
