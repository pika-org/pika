//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/preprocessor/nargs.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/assertion/current_function.hpp>
#include <pika/assertion/evaluate_assert.hpp>
#include <pika/assertion/source_location.hpp>

#if defined(PIKA_COMPUTE_DEVICE_CODE)
# include <assert.h>
#endif

#include <exception>
#include <string>
#include <type_traits>
#endif

# define PIKA_ASSERT_CURRENT_FUNCTION __PRETTY_FUNCTION__

#if defined(DOXYGEN)
/// \def PIKA_ASSERT(expr, msg)
/// \brief This macro asserts that \a expr evaluates to true.
///
/// \param expr The expression to assert on. This can either be an expression
///             that's convertible to bool or a callable which returns bool
/// \param msg The optional message that is used to give further information if
///             the assert fails. This should be convertible to a std::string
///
/// If \p expr evaluates to false, The source location and \p msg is being
/// printed along with the expression and additional. Afterwards the program is
/// being aborted. The assertion handler can be customized by calling
/// pika::detail::set_assertion_handler().
///
/// Asserts are enabled if \a PIKA_DEBUG is set. This is the default for
/// `CMAKE_BUILD_TYPE=Debug`
# define PIKA_ASSERT(expr)

/// \see PIKA_ASSERT
# define PIKA_ASSERT_MSG(expr, msg)
#else
/// \cond NOINTERNAL
# define PIKA_ASSERT_(expr, msg)                                                                   \
     (!!(expr) ? void() :                                                                          \
                 ::pika::detail::handle_assert(                                                    \
                     ::pika::detail::source_location{                                              \
                         __FILE__, static_cast<unsigned>(__LINE__), PIKA_ASSERT_CURRENT_FUNCTION}, \
                     PIKA_PP_STRINGIZE(expr), msg)) /**/

# if defined(PIKA_DEBUG)
#  if defined(PIKA_COMPUTE_DEVICE_CODE)
#   define PIKA_ASSERT(expr) assert(expr)
#   define PIKA_ASSERT_MSG(expr, msg) PIKA_ASSERT(expr)
#  else
#   define PIKA_ASSERT(expr) PIKA_ASSERT_(expr, std::string())
#   define PIKA_ASSERT_MSG(expr, msg) PIKA_ASSERT_(expr, msg)
#  endif
# else
#  define PIKA_ASSERT(expr)
#  define PIKA_ASSERT_MSG(expr, msg)
# endif

# define PIKA_UNREACHABLE                                                                          \
     PIKA_ASSERT_(false,                                                                           \
         "This code is meant to be unreachable. If you are seeing this error "                     \
         "message it means that you have found a bug in pika. Please report "                      \
         "it on https://github.com/pika-org/pika/issues.");                                        \
     std::terminate()
#endif
