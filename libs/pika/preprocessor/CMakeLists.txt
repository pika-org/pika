# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/preprocessor/include
set(preprocessor_headers
    pika/preprocessor/cat.hpp
    pika/preprocessor/config.hpp
    pika/preprocessor/expand.hpp
    pika/preprocessor/identity.hpp
    pika/preprocessor/nargs.hpp
    pika/preprocessor/stringize.hpp
    pika/preprocessor/strip_parens.hpp
)

# Default location is $PIKA_ROOT/libs/preprocessor/src
set(preprocessor_sources)

include(pika_add_module)
pika_add_module(
  pika preprocessor
  SOURCES ${preprocessor_sources}
  HEADERS ${preprocessor_headers}
  CMAKE_SUBDIRS tests
)
