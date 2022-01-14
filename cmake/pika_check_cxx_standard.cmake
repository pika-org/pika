# Copyright (c) 2020 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least C++17. However, if a higher standard is set by the user in
# CMAKE_CXX_STANDARD that requirement has to be propagated to users of pika
# as well (i.e. pika can't be compiled with C++20 and applications with
# C++17; the other way around is allowed). Ideally, users should not set
# CMAKE_CXX_STANDARD when building pika.
pika_option(
  PIKA_WITH_CXX_STANDARD STRING
  "C++ standard to use for compiling pika (default: 17)" "17" ADVANCED
)

if(PIKA_WITH_CXX_STANDARD LESS 17)
  pika_error(
    "You've set PIKA_WITH_CXX_STANDARD to ${PIKA_WITH_CXX_STANDARD}, which is less than 17 which is the minimum required by pika"
  )
endif()

if(DEFINED CMAKE_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD STREQUAL
                                  PIKA_WITH_CXX_STANDARD
)
  pika_error(
    "You've set CMAKE_CXX_STANDARD to ${CMAKE_CXX_STANDARD} and PIKA_WITH_CXX_STANDARD to ${PIKA_WITH_CXX_STANDARD}. Please unset CMAKE_CXX_STANDARD."
  )
endif()

set(CMAKE_CXX_STANDARD ${PIKA_WITH_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# We explicitly set the default to 98 to force CMake to emit a -std=c++XX flag.
# Some compilers (clang) have a different default standard for cpp and cu files,
# but CMake does not know about this difference. If the standard is set to the
# .cpp default in CMake, CMake will omit the flag, resulting in the wrong
# standard for .cu files.
set(CMAKE_CXX_STANDARD_DEFAULT 98)

pika_info("Using C++${PIKA_WITH_CXX_STANDARD}")
