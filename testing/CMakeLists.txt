# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(testing_sources testing.cpp performance.cpp)
list(TRANSFORM testing_sources PREPEND src/ OUTPUT_VARIABLE testing_sources)

add_library(pika_testing SHARED ${testing_sources})
target_include_directories(
  pika_testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pika_testing PRIVATE pika_private_flags)
target_link_libraries(pika_testing PUBLIC pika::pika)
