# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(pika_add_library name)
  # retrieve arguments
  set(options
      EXCLUDE_FROM_ALL
      INTERNAL_FLAGS
      NOLIBS
      NOEXPORT
      NOPCH
      OBJECT
      NONAMEPREFIX
      UNITY_BUILD
  )
  set(one_value_args
      FOLDER
      SOURCE_ROOT
      HEADER_ROOT
      SOURCE_GLOB
      HEADER_GLOB
      OUTPUT_SUFFIX
      INSTALL_SUFFIX
  )
  set(multi_value_args SOURCES HEADERS AUXILIARY DEPENDENCIES COMPILER_FLAGS LINK_FLAGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  pika_debug("add_library.${name}" "${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  pika_debug("add_library.${name}" "${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

  pika_add_library_sources_noglob(${name} SOURCES "${${name}_SOURCES}")

  pika_add_source_group(
    NAME ${name}
    CLASS "Source Files"
    ROOT ${${name}_SOURCE_ROOT}
    TARGETS ${${name}_SOURCES}
  )

  pika_add_library_headers_noglob(${name} HEADERS "${${name}_HEADERS}")

  pika_add_source_group(
    NAME ${name}
    CLASS "Header Files"
    ROOT ${${name}_HEADER_ROOT}
    TARGETS ${${name}_HEADERS}
  )

  pika_print_list("DEBUG" "add_library.${name}" "Sources for ${name}" ${name}_SOURCES)
  pika_print_list("DEBUG" "add_library.${name}" "Headers for ${name}" ${name}_HEADERS)
  pika_print_list("DEBUG" "add_library.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)

  set(exclude_from_all)

  if(${name}_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  else()
    if(MSVC)
      set(library_install_destination ${CMAKE_INSTALL_BINDIR})
    else()
      set(library_install_destination ${CMAKE_INSTALL_LIBDIR})
    endif()
    set(archive_install_destination ${CMAKE_INSTALL_LIBDIR})
    set(runtime_install_destination ${CMAKE_INSTALL_BINDIR})
    if(${name}_INSTALL_SUFFIX)
      set(library_install_destination ${${name}_INSTALL_SUFFIX})
      set(archive_install_destination ${${name}_INSTALL_SUFFIX})
      set(runtime_install_destination ${${name}_INSTALL_SUFFIX})
    endif()
    # cmake-format: off
    set(_target_flags
        INSTALL INSTALL_FLAGS
          LIBRARY DESTINATION ${library_install_destination}
          ARCHIVE DESTINATION ${archive_install_destination}
          RUNTIME DESTINATION ${runtime_install_destination}
    )
    # cmake-format: on

    # install PDB if needed
    if(MSVC AND BUILD_SHARED_LIBS)
      # cmake-format: off
      set(_target_flags
          ${_target_flags}
          INSTALL_PDB $<TARGET_PDB_FILE:${name}>
            DESTINATION ${runtime_install_destination}
          CONFIGURATIONS Debug RelWithDebInfo
          OPTIONAL
      )
      # cmake-format: on
    endif()
  endif()

  if(${name}_NONAMEPREFIX)
    set(_target_flags ${_target_flags} NONAMEPREFIX)
  endif()

  set(${name}_linktype)
  if(${name}_OBJECT)
    set(${name}_linktype OBJECT)
  endif()

  if(PIKA_WITH_HIP)
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        pika_add_nvhpc_cuda_flags(${source})
      endif()
    endforeach()
  endif()

  add_library(
    ${name} ${${name}_linktype} ${exclude_from_all} ${${name}_SOURCES} ${${name}_HEADERS}
            ${${name}_AUXILIARY}
  )

  if(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(
        ${name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE
                   "${PIKA_WITH_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_RELEASE
                   "${PIKA_WITH_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_RELEASE
                   "${PIKA_WITH_BINARY_DIR}/Release/lib/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_DEBUG
                   "${PIKA_WITH_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_DEBUG
                   "${PIKA_WITH_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_DEBUG
                   "${PIKA_WITH_BINARY_DIR}/Debug/lib/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
                   "${PIKA_WITH_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL
                   "${PIKA_WITH_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL
                   "${PIKA_WITH_BINARY_DIR}/MinSizeRel/lib/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${PIKA_WITH_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${PIKA_WITH_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${PIKA_WITH_BINARY_DIR}/RelWithDebInfo/lib/${${name}_OUTPUT_SUFFIX}"
      )
    else()
      set_target_properties(
        ${name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PIKA_WITH_BINARY_DIR}/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY "${PIKA_WITH_BINARY_DIR}/lib/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY "${PIKA_WITH_BINARY_DIR}/lib/${${name}_OUTPUT_SUFFIX}"
      )
    endif()
  endif()

  # get public and private compile options that pika needs
  if(${${name}_NOLIBS})
    set(_target_flags ${_target_flags} NOLIBS)
  endif()

  if(NOT ${${name}_NOEXPORT})
    set(_target_flags ${_target_flags} EXPORT)
  endif()

  if(${${name}_NOPCH})
    set(_target_flags ${_target_flags} NOPCH)
  endif()

  if(${name}_INTERNAL_FLAGS)
    set(_target_flags ${_target_flags} INTERNAL_FLAGS)
  endif()

  if(${name}_UNITY_BUILD)
    set(_target_flags ${_target_flags} UNITY_BUILD)
  endif()

  pika_setup_target(
    ${name}
    TYPE LIBRARY
    NAME ${name}
    FOLDER ${${name}_FOLDER}
    COMPILE_FLAGS ${${name}_COMPILE_FLAGS}
    LINK_FLAGS ${${name}_LINK_FLAGS}
    DEPENDENCIES ${${name}_DEPENDENCIES}
  )

endfunction()
