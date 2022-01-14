//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/defines.hpp>

#if defined(DOXYGEN)
/// Marks a class or function to be exported from pika or imported if it is
/// consumed.
#define PIKA_EXPORT
#else

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#if !defined(PIKA_MODULE_STATIC_LINKING)
#define PIKA_SYMBOL_EXPORT __declspec(dllexport)
#define PIKA_SYMBOL_IMPORT __declspec(dllimport)
#define PIKA_SYMBOL_INTERNAL /* empty */
#endif
#elif defined(__NVCC__) || defined(__CUDACC__)
#define PIKA_SYMBOL_EXPORT   /* empty */
#define PIKA_SYMBOL_IMPORT   /* empty */
#define PIKA_SYMBOL_INTERNAL /* empty */
#elif defined(PIKA_HAVE_ELF_HIDDEN_VISIBILITY)
#define PIKA_SYMBOL_EXPORT __attribute__((visibility("default")))
#define PIKA_SYMBOL_IMPORT __attribute__((visibility("default")))
#define PIKA_SYMBOL_INTERNAL __attribute__((visibility("hidden")))
#endif

// make sure we have reasonable defaults
#if !defined(PIKA_SYMBOL_EXPORT)
#define PIKA_SYMBOL_EXPORT /* empty */
#endif
#if !defined(PIKA_SYMBOL_IMPORT)
#define PIKA_SYMBOL_IMPORT /* empty */
#endif
#if !defined(PIKA_SYMBOL_INTERNAL)
#define PIKA_SYMBOL_INTERNAL /* empty */
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(PIKA_EXPORTS)
#define PIKA_EXPORT PIKA_SYMBOL_EXPORT
#else
#define PIKA_EXPORT PIKA_SYMBOL_IMPORT
#endif

///////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
#define PIKA_ALWAYS_EXPORT PIKA_SYMBOL_EXPORT
#define PIKA_ALWAYS_IMPORT PIKA_SYMBOL_IMPORT
#endif
