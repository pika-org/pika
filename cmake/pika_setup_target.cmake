# Copyright (c) 2014      Thomas Heller
# Copyright (c) 2007-2018 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_policy(PUSH)

include(pika_generate_package_utils)

pika_set_cmake_policy(CMP0054 NEW)
pika_set_cmake_policy(CMP0060 NEW)

function(pika_setup_target target)
  set(options
      EXPORT
      INSTALL
      INSTALL_HEADERS
      INTERNAL_FLAGS
      NOLIBS
      NONAMEPREFIX
      NOTLLKEYWORD
      UNITY_BUILD
  )
  set(one_value_args
      TYPE
      FOLDER
      NAME
      SOVERSION
      VERSION
      HEADER_ROOT
  )
  set(multi_value_args DEPENDENCIES COMPILE_FLAGS LINK_FLAGS INSTALL_FLAGS
                       INSTALL_PDB
  )
  cmake_parse_arguments(
    target "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT TARGET ${target})
    pika_error("${target} does not represent a target")
  endif()

  # Figure out which type we want...
  if(target_TYPE)
    string(TOUPPER "${target_TYPE}" _type)
  else()
    pika_get_target_property(type_prop ${target} TYPE)
    if(type_prop STREQUAL "STATIC_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "MODULE_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "SHARED_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "EXECUTABLE")
      set(_type "EXECUTABLE")
    endif()
  endif()

  if(target_FOLDER)
    set_target_properties(${target} PROPERTIES FOLDER "${target_FOLDER}")
  endif()

  pika_get_target_property(target_SOURCES ${target} SOURCES)

  if(target_COMPILE_FLAGS)
    pika_append_property(${target} COMPILE_FLAGS ${target_COMPILE_FLAGS})
  endif()

  if(target_LINK_FLAGS)
    pika_append_property(${target} LINK_FLAGS ${target_LINK_FLAGS})
  endif()

  if(target_NAME)
    set(name "${target_NAME}")
  else()
    set(name "${target}")
  endif()

  if(target_NOTLLKEYWORD)
    set(__tll_private)
    set(__tll_public)
  else()
    set(__tll_private PRIVATE)
    set(__tll_public PUBLIC)
  endif()

  set(target_STATIC_LINKING OFF)
  set(_pika_library_type)
  if(TARGET pika)
    pika_get_target_property(_pika_library_type pika TYPE)
  endif()

  if("${_pika_library_type}" STREQUAL "STATIC_LIBRARY")
    set(target_STATIC_LINKING ON)
  endif()

  if("${_type}" STREQUAL "EXECUTABLE")
    target_compile_definitions(
      ${target} PRIVATE "PIKA_APPLICATION_NAME=${name}"
                        "PIKA_APPLICATION_STRING=\"${name}\""
    )
  endif()

  if("${_type}" STREQUAL "LIBRARY")
    if(DEFINED PIKA_LIBRARY_VERSION AND DEFINED PIKA_SOVERSION)
      # set properties of generated shared library
      set_target_properties(
        ${target} PROPERTIES VERSION ${PIKA_LIBRARY_VERSION}
                             SOVERSION ${PIKA_SOVERSION}
      )
    endif()
    if(NOT target_NONAMEPREFIX)
      pika_set_lib_name(${target} ${name})
    endif()
    set_target_properties(
      ${target}
      PROPERTIES # create *nix style library versions + symbolic links
                 # allow creating static and shared libs without conflicts
                 CLEAN_DIRECT_OUTPUT 1 OUTPUT_NAME ${name}
    )
  endif()

  if(NOT target_NOLIBS)
    target_link_libraries(${target} ${__tll_public} pika::pika)
    if(PIKA_WITH_PRECOMPILED_HEADERS_INTERNAL)
      if("${_type}" STREQUAL "EXECUTABLE")
        target_precompile_headers(
          ${target} REUSE_FROM pika_exe_precompiled_headers
        )
      endif()
    endif()
  endif()

  target_link_libraries(${target} ${__tll_public} ${target_DEPENDENCIES})

  if(target_INTERNAL_FLAGS AND TARGET pika_private_flags)
    target_link_libraries(${target} ${__tll_private} pika_private_flags)
  endif()

  if(target_UNITY_BUILD)
    set_target_properties(${target} PROPERTIES UNITY_BUILD ON)
  endif()

  get_target_property(target_EXCLUDE_FROM_ALL ${target} EXCLUDE_FROM_ALL)

  if(target_EXPORT AND NOT target_EXCLUDE_FROM_ALL)
    pika_export_targets(${target})
    set(install_export EXPORT pika_targets)
  endif()

  if(target_INSTALL AND NOT target_EXCLUDE_FROM_ALL)
    install(TARGETS ${target} ${install_export} ${target_INSTALL_FLAGS})
    if(target_INSTALL_PDB)
      install(FILES ${target_INSTALL_PDB})
    endif()
    if(target_INSTALL_HEADERS AND (NOT target_HEADER_ROOT STREQUAL ""))
      install(
        DIRECTORY "${target_HEADER_ROOT}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT ${name}
      )
    endif()
  endif()
endfunction()

cmake_policy(POP)
