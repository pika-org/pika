//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file throw_exception.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/assertion/current_function.hpp>
#include <pika/errors/error.hpp>
#include <pika/errors/exception_fwd.hpp>
#include <pika/modules/format.hpp>
#include <pika/preprocessor/cat.hpp>
#include <pika/preprocessor/expand.hpp>
#include <pika/preprocessor/nargs.hpp>

#include <exception>
#include <string>
#include <system_error>

#include <pika/local/config/warnings_prefix.hpp>

/// \cond NODETAIL
namespace pika { namespace detail {
    template <typename Exception>
    PIKA_NORETURN PIKA_EXPORT void throw_exception(Exception const& e,
        std::string const& func, std::string const& file, long line);

    PIKA_NORETURN PIKA_EXPORT void throw_exception(error errcode,
        std::string const& msg, std::string const& func,
        std::string const& file, long line);

    PIKA_NORETURN PIKA_EXPORT void rethrow_exception(
        exception const& e, std::string const& func);

    template <typename Exception>
    PIKA_EXPORT std::exception_ptr get_exception(Exception const& e,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    PIKA_EXPORT std::exception_ptr get_exception(error errcode,
        std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    PIKA_EXPORT std::exception_ptr get_exception(std::error_code const& ec,
        std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    PIKA_EXPORT void throws_if(pika::error_code& ec, error errcode,
        std::string const& msg, std::string const& func,
        std::string const& file, long line);

    PIKA_EXPORT void rethrows_if(
        pika::error_code& ec, exception const& e, std::string const& func);

    PIKA_NORETURN PIKA_EXPORT void throw_thread_interrupted_exception();
}}    // namespace pika::detail
/// \endcond

namespace pika {
    /// \cond NOINTERNAL

    /// \brief throw an pika::exception initialized from the given arguments
    PIKA_NORETURN inline void throw_exception(error e, std::string const& msg,
        std::string const& func, std::string const& file = "", long line = -1)
    {
        detail::throw_exception(e, msg, func, file, line);
    }
    /// \endcond
}    // namespace pika

/// \cond NOINTERNAL
///////////////////////////////////////////////////////////////////////////////
// helper macro allowing to prepend file name and line number to a generated
// exception
#define PIKA_THROW_STD_EXCEPTION(except, func)                                  \
    pika::detail::throw_exception(except, func, __FILE__, __LINE__) /**/

#define PIKA_RETHROW_EXCEPTION(e, f) pika::detail::rethrow_exception(e, f) /**/

#define PIKA_RETHROWS_IF(ec, e, f) pika::detail::rethrows_if(ec, e, f) /**/

///////////////////////////////////////////////////////////////////////////////
#define PIKA_GET_EXCEPTION(...)                                                 \
    PIKA_GET_EXCEPTION_(__VA_ARGS__)                                            \
    /**/

#define PIKA_GET_EXCEPTION_(...)                                                \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_GET_EXCEPTION_, PIKA_PP_NARGS(__VA_ARGS__))(   \
        __VA_ARGS__))                                                          \
/**/
#define PIKA_GET_EXCEPTION_3(errcode, f, msg)                                   \
    PIKA_GET_EXCEPTION_4(errcode, pika::plain, f, msg)                           \
/**/
#define PIKA_GET_EXCEPTION_4(errcode, mode, f, msg)                             \
    pika::detail::get_exception(errcode, msg, mode, f, __FILE__, __LINE__) /**/

///////////////////////////////////////////////////////////////////////////////
#define PIKA_THROW_IN_CURRENT_FUNC(errcode, msg)                                \
    PIKA_THROW_EXCEPTION(errcode, PIKA_ASSERTION_CURRENT_FUNCTION, msg)          \
    /**/

#define PIKA_RETHROW_IN_CURRENT_FUNC(errcode, msg)                              \
    PIKA_RETHROW_EXCEPTION(errcode, PIKA_ASSERTION_CURRENT_FUNCTION, msg)        \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define PIKA_THROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                        \
    PIKA_THROWS_IF(ec, errcode, PIKA_ASSERTION_CURRENT_FUNCTION, msg)            \
    /**/

#define PIKA_RETHROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                      \
    PIKA_RETHROWS_IF(ec, errcode, PIKA_ASSERTION_CURRENT_FUNCTION, msg)          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define PIKA_THROW_THREAD_INTERRUPTED_EXCEPTION()                               \
    pika::detail::throw_thread_interrupted_exception() /**/
/// \endcond

///////////////////////////////////////////////////////////////////////////////
/// \def PIKA_THROW_EXCEPTION(errcode, f, msg)
/// \brief Throw a pika::exception initialized from the given parameters
///
/// The macro \a PIKA_THROW_EXCEPTION can be used to throw a pika::exception.
/// The purpose of this macro is to prepend the source file name and line number
/// of the position where the exception is thrown to the error message.
/// Moreover, this associates additional diagnostic information with the
/// exception, such as file name and line number, locality id and thread id,
/// and stack backtrace from the point where the exception was thrown.
///
/// The parameter \p errcode holds the pika::error code the new exception should
/// encapsulate. The parameter \p f is expected to hold the name of the
/// function exception is thrown from and the parameter \p msg holds the error
/// message the new exception should encapsulate.
///
/// \par Example:
///
/// \code
///      void raise_exception()
///      {
///          // Throw a pika::exception initialized from the given parameters.
///          // Additionally associate with this exception some detailed
///          // diagnostic information about the throw-site.
///          PIKA_THROW_EXCEPTION(pika::no_success, "raise_exception", "simulated error");
///      }
/// \endcode
///
#define PIKA_THROW_EXCEPTION(errcode, f, ...)                                   \
    pika::detail::throw_exception(                                              \
        errcode, pika::util::format(__VA_ARGS__), f, __FILE__, __LINE__) /**/

/// \def PIKA_THROWS_IF(ec, errcode, f, msg)
/// \brief Either throw a pika::exception or initialize \a pika::error_code from
///        the given parameters
///
/// The macro \a PIKA_THROWS_IF can be used to either throw a \a pika::exception
/// or to initialize a \a pika::error_code from the given parameters. If
/// &ec == &pika::throws, the semantics of this macro are equivalent to
/// \a PIKA_THROW_EXCEPTION. If &ec != &pika::throws, the \a pika::error_code
/// instance \p ec is initialized instead.
///
/// The parameter \p errcode holds the pika::error code from which the new
/// exception should be initialized. The parameter \p f is expected to hold the
/// name of the function exception is thrown from and the parameter \p msg
/// holds the error message the new exception should encapsulate.
///
#define PIKA_THROWS_IF(ec, errcode, f, ...)                                     \
    pika::detail::throws_if(ec, errcode, pika::util::format(__VA_ARGS__), f,     \
        __FILE__, __LINE__) /**/

#include <pika/local/config/warnings_suffix.hpp>
