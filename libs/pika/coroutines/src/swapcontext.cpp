//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>

#if defined(PIKA_HAVE_MODULE)
module pika.coroutines;
#endif

#if !defined(PIKA_HAVE_BOOST_CONTEXT)

# if (defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)) &&         \
     !defined(__bgq__) && !defined(__powerpc__) && !defined(__s390x__) && !defined(__arm__) &&     \
     !defined(__arm64__) && !defined(__aarch64__)

#  if defined(__x86_64__) || defined(__amd64__)
#   include "swapcontext64.ipp"
#  elif defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__)
#   include "swapcontext32.ipp"
#  else
#   error You are trying to use x86 context switching on a non-x86 platform. Your \
    platform may be supported with the CMake option \
    PIKA_WITH_BOOST_CONTEXT=ON (requires Boost.Context).
#  endif

# endif

#endif
