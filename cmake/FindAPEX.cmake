# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET pika_internal::apex)
  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_APEX QUIET apex)

  find_path(
    APEX_INCLUDE_DIR task_wrapper.hpp
    HINTS ENV APEX_ROOT ${PIKA_APEX_ROOT} ${PC_APEX_INCLUDEDIR} ${PC_APEX_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    APEX_LIBRARY
    NAMES apex libapex
    HINTS ${APEX_ROOT}
          ENV
          APEX_ROOT
          ${PIKA_APEX_ROOT}
          ${PC_APEX_MINIMAL_LIBDIR}
          ${PC_APEX_MINIMAL_LIBRARY_DIRS}
          ${PC_APEX_LIBDIR}
          ${PC_APEX_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  # Set APEX_ROOT in case the other hints are used
  if(APEX_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${APEX_ROOT} APEX_ROOT)
  elseif("$ENV{APEX_ROOT}")
    file(TO_CMAKE_PATH $ENV{APEX_ROOT} APEX_ROOT)
  else()
    file(TO_CMAKE_PATH "${APEX_INCLUDE_DIR}" APEX_INCLUDE_DIR)
    string(REPLACE "/include" "" APEX_ROOT "${APEX_INCLUDE_DIR}")
  endif()

  set(APEX_LIBRARIES ${APEX_LIBRARY})
  set(APEX_INCLUDE_DIRS ${APEX_INCLUDE_DIR})

  find_package_handle_standard_args(APEX DEFAULT_MSG APEX_LIBRARY APEX_INCLUDE_DIR)

  get_property(
    _type
    CACHE APEX_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE APEX_ROOT PROPERTY ADVANCED 1)
    if(_type STREQUAL "UNINITIALIZED")
      set_property(CACHE APEX_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  add_library(pika_internal::apex INTERFACE IMPORTED)
  target_include_directories(pika_internal::apex SYSTEM INTERFACE ${APEX_INCLUDE_DIR})
  target_link_libraries(pika_internal::apex INTERFACE ${APEX_LIBRARIES})

  mark_as_advanced(APEX_ROOT APEX_LIBRARY APEX_INCLUDE_DIR)
endif()
