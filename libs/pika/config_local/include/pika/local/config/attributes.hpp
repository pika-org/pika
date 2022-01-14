//  Copyright (c) 2017 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/defines.hpp>
#include <pika/local/config/compiler_specific.hpp>

#if defined(DOXYGEN)

/// Function attribute to tell compiler not to inline the function.
#define PIKA_NOINLINE

/// Function attribute to tell compiler that the function does not return.
#define PIKA_NORETURN

/// Marks an entity as deprecated. The argument \c x specifies a custom message
/// that is included in the compiler warning. For more details see
/// `<https://en.cppreference.com/w/cpp/language/attributes/deprecated>`__.
#define PIKA_DEPRECATED(x)

/// Indicates that the fall through from the previous case label is intentional
/// and should not be diagnosed by a compiler that warns on fallthrough. For
/// more details see
/// `<https://en.cppreference.com/w/cpp/language/attributes/fallthrough>`__.
#define PIKA_FALLTHROUGH

/// If a function declared nodiscard or a function returning an enumeration or
/// class declared nodiscard by value is called from a discarded-value expression
/// other than a cast to void, the compiler is encouraged to issue a warning.
/// For more details see
/// `https://en.cppreference.com/w/cpp/language/attributes/nodiscard`__.
#define PIKA_NODISCARD

/// Indicates that this data member need not have an address distinct from all
/// other non-static data members of its class.
/// For more details see
/// `https://en.cppreference.com/w/cpp/language/attributes/no_unique_address`__.
#define PIKA_NO_UNIQUE_ADDRESS
#else

///////////////////////////////////////////////////////////////////////////////
// clang-format off
#if defined(PIKA_MSVC)
#   define PIKA_NOINLINE __declspec(noinline)
#elif defined(__GNUC__)
#   if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
        // nvcc doesn't always parse __noinline
#       define PIKA_NOINLINE __attribute__ ((noinline))
#   else
#       define PIKA_NOINLINE __attribute__ ((__noinline__))
#   endif
#else
#   define PIKA_NOINLINE
#endif

///////////////////////////////////////////////////////////////////////////////
// handle [[noreturn]]
#define PIKA_NORETURN [[noreturn]]

///////////////////////////////////////////////////////////////////////////////
// handle [[deprecated]]
#if PIKA_HAVE_DEPRECATION_WARNINGS && !defined(PIKA_INTEL_VERSION)
#  define PIKA_DEPRECATED_MSG \
   "This functionality is deprecated and will be removed in the future."
#  define PIKA_DEPRECATED(x) [[deprecated(x)]]
#endif

#if !defined(PIKA_DEPRECATED)
#  define PIKA_DEPRECATED(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// handle [[fallthrough]]
#define PIKA_FALLTHROUGH [[fallthrough]]

///////////////////////////////////////////////////////////////////////////////
// handle empty_bases
#if defined(_MSC_VER)
#  define PIKA_EMPTY_BASES __declspec(empty_bases)
#else
#  define PIKA_EMPTY_BASES
#endif

///////////////////////////////////////////////////////////////////////////////
// handle [[nodiscard]]
#define PIKA_NODISCARD [[nodiscard]]
#define PIKA_NODISCARD_MSG(x) [[nodiscard(x)]]

///////////////////////////////////////////////////////////////////////////////
// handle [[no_unique_address]]
#if defined(PIKA_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE)
#   define PIKA_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#   define PIKA_NO_UNIQUE_ADDRESS
#endif

///////////////////////////////////////////////////////////////////////////////
// handle empty_bases
#if defined(_MSC_VER)
#  define PIKA_EMPTY_BASES __declspec(empty_bases)
#else
#  define PIKA_EMPTY_BASES
#endif

// clang-format on

#endif
