# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION AND NOT PIKA_FIND_PACKAGE)
  if(PIKA_WITH_CXX_STANDARD LESS 20)
    pika_error(
      "PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION requires at least C++20 (PIKA_WITH_CXX_STANDARD is currently ${PIKA_WITH_CXX_STANDARD})"
    )
  endif()

  include(FetchContent)

  fetchcontent_declare(
    P2300
    GIT_REPOSITORY https://github.com/brycelelbach/wg21_P2300_std_execution.git
    GIT_TAG 6da4968c0093c188f7d4dfce75a9d401944d8f06
  )
  fetchcontent_getproperties(P2300)
  if(NOT p2300_POPULATED)
    fetchcontent_populate(P2300)
  endif()

  add_library(P2300 INTERFACE)
  target_include_directories(
    P2300 SYSTEM INTERFACE "$<BUILD_INTERFACE:${p2300_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:include>"
  )
  target_compile_features(P2300 INTERFACE cxx_std_20)

  install(
    TARGETS P2300
    EXPORT pika_internal_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT core
  )
  pika_export_internal_targets(P2300)

  install(
    DIRECTORY "${p2300_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT core
  )

  target_link_libraries(pika_base_libraries INTERFACE P2300)
endif()
