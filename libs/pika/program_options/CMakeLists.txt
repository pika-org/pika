# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/program_options/include
set(program_options_headers
    pika/program_options.hpp
    pika/modules/program_options.hpp
    pika/program_options/cmdline.hpp
    pika/program_options/config.hpp
    pika/program_options/environment_iterator.hpp
    pika/program_options/eof_iterator.hpp
    pika/program_options/errors.hpp
    pika/program_options/option.hpp
    pika/program_options/options_description.hpp
    pika/program_options/parsers.hpp
    pika/program_options/positional_options.hpp
    pika/program_options/value_semantic.hpp
    pika/program_options/variables_map.hpp
    pika/program_options/version.hpp
    pika/program_options/detail/cmdline.hpp
    pika/program_options/detail/config_file.hpp
    pika/program_options/detail/convert.hpp
    pika/program_options/detail/parsers.hpp
    pika/program_options/detail/utf8_codecvt_facet.hpp
    pika/program_options/detail/value_semantic.hpp
)

# Default location is $PIKA_ROOT/libs/program_options/src
set(program_options_sources
    cmdline.cpp
    config_file.cpp
    convert.cpp
    options_description.cpp
    parsers.cpp
    positional_options.cpp
    split.cpp
    utf8_codecvt_facet.cpp
    value_semantic.cpp
    variables_map.cpp
    winmain.cpp
)

include(pika_add_module)
pika_add_module(
  pika program_options
  GLOBAL_HEADER_GEN OFF
  SOURCES ${program_options_sources}
  HEADERS ${program_options_headers}
  DEPENDENCIES ${__boost_program_options}
  MODULE_DEPENDENCIES pika_config_local pika_datastructures pika_format
                      pika_iterator_support
  CMAKE_SUBDIRS examples tests
)
