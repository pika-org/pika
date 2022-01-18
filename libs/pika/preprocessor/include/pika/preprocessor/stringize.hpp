//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the pika Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/* Copyright (C) 2001
 * Housemarque Oy
 * http://www.housemarque.com
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

/* Revised by Paul Mensonides (2002) */

/* See http://www.boost.org for most recent version. */

// pikainspect:noinclude:PIKA_PP_STRINGIZE

#pragma once

#if defined(DOXYGEN)
/// The \a PIKA_PP_STRINGIZE macro stringizes its argument after it has been expanded.
///
/// \param X The text to be converted to a string literal
///
/// The passed argument \c X will expand to \c "X". Note that the stringizing
/// operator (#) prevents arguments from expanding. This macro circumvents this
/// shortcoming.
#define PIKA_PP_STRINGIZE(X)
#else

#include <pika/preprocessor/config.hpp>

#if PIKA_PP_CONFIG_FLAGS() & PIKA_PP_CONFIG_MSVC()
#define PIKA_PP_STRINGIZE(text) PIKA_PP_STRINGIZE_A((text))
#define PIKA_PP_STRINGIZE_A(arg) PIKA_PP_STRINGIZE_I arg
#elif PIKA_PP_CONFIG_FLAGS() & PIKA_PP_CONFIG_MWCC()
#define PIKA_PP_STRINGIZE(text) PIKA_PP_STRINGIZE_OO((text))
#define PIKA_PP_STRINGIZE_OO(par) PIKA_PP_STRINGIZE_I##par
#else
#define PIKA_PP_STRINGIZE(text) PIKA_PP_STRINGIZE_I(text)
#endif

#define PIKA_PP_STRINGIZE_I(text) #text

#endif
