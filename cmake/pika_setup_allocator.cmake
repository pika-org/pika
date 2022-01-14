# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2013 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(pika_add_definitions)

# In case find_package(pika) is called multiple times
if(NOT TARGET pika_dependencies_allocator)

  if(NOT PIKA_WITH_MALLOC)
    set(PIKA_WITH_MALLOC
        CACHE
          STRING
          "Use the specified allocator. Supported allocators are tcmalloc, jemalloc, tbbmalloc and system."
          ${DEFAULT_MALLOC}
    )
    set(allocator_error
        "The default allocator for your system is ${DEFAULT_MALLOC}, but ${DEFAULT_MALLOC} could not be found. "
        "The system allocator has poor performance. As such ${DEFAULT_MALLOC} is a strong optional requirement. "
        "Being aware of the performance hit, you can override this default and get rid of this dependency by setting -DPIKA_WITH_MALLOC=system. "
        "Valid options for PIKA_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom"
    )
  else()
    set(allocator_error
        "PIKA_WITH_MALLOC was set to ${PIKA_WITH_MALLOC}, but ${PIKA_WITH_MALLOC} could not be found. "
        "Valid options for PIKA_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom"
    )
  endif()

  string(TOUPPER "${PIKA_WITH_MALLOC}" PIKA_WITH_MALLOC_UPPER)

  add_library(pika_dependencies_allocator INTERFACE IMPORTED)

  if(NOT PIKA_WITH_MALLOC_DEFAULT)

    # ##########################################################################
    # TCMALLOC
    if("${PIKA_WITH_MALLOC_UPPER}" STREQUAL "TCMALLOC")
      find_package(TCMalloc)
      if(NOT TCMALLOC_LIBRARIES)
        pika_error(${allocator_error})
      endif()

      target_link_libraries(
        pika_dependencies_allocator INTERFACE ${TCMALLOC_LIBRARIES}
      )

      if(MSVC)
        target_compile_options(
          pika_dependencies_allocator INTERFACE /INCLUDE:__tcmalloc
        )
      endif()
      set(_use_custom_allocator TRUE)
    endif()

    # ##########################################################################
    # JEMALLOC
    if("${PIKA_WITH_MALLOC_UPPER}" STREQUAL "JEMALLOC")
      find_package(Jemalloc)
      if(NOT JEMALLOC_LIBRARIES)
        pika_error(${allocator_error})
      endif()
      target_include_directories(
        pika_dependencies_allocator
        INTERFACE ${JEMALLOC_INCLUDE_DIR} ${JEMALLOC_ADDITIONAL_INCLUDE_DIR}
      )
      target_link_libraries(
        pika_dependencies_allocator INTERFACE ${JEMALLOC_LIBRARIES}
      )
    endif()

    # ##########################################################################
    # MIMALLOC
    if("${PIKA_WITH_MALLOC_UPPER}" STREQUAL "MIMALLOC")
      find_package(mimalloc 1.0)
      if(NOT mimalloc_FOUND)
        pika_error(${allocator_error})
      endif()
      target_link_libraries(pika_dependencies_allocator INTERFACE mimalloc)
      set(pika_MALLOC_LIBRARY mimalloc)
      if(MSVC)
        target_compile_options(
          pika_dependencies_allocator INTERFACE /INCLUDE:mi_version
        )
      endif()
      set(_use_custom_allocator TRUE)
    endif()

    # ##########################################################################
    # TBBMALLOC
    if("${PIKA_WITH_MALLOC_UPPER}" STREQUAL "TBBMALLOC")
      find_package(TBBmalloc)
      if(NOT TBBMALLOC_LIBRARY AND NOT TBBMALLOC_PROXY_LIBRARY)
        pika_error(${allocator_error})
      endif()
      if(MSVC)
        target_compile_options(
          pika_dependencies_allocator
          INTERFACE /INCLUDE:__TBB_malloc_proxy
        )
      endif()
      target_link_libraries(
        pika_dependencies_allocator INTERFACE ${TBBMALLOC_LIBRARY}
                                                   ${TBBMALLOC_PROXY_LIBRARY}
      )
    endif()

    if("${PIKA_WITH_MALLOC_UPPER}" STREQUAL "CUSTOM")
      set(_use_custom_allocator TRUE)
    endif()

  else()

    set(PIKA_WITH_MALLOC ${PIKA_WITH_MALLOC_DEFAULT})

  endif(NOT PIKA_WITH_MALLOC_DEFAULT)

  if("${PIKA_WITH_MALLOC_UPPER}" MATCHES "SYSTEM")
    if(NOT MSVC)
      pika_warn(
        "pika will perform poorly without tcmalloc, jemalloc, or mimalloc. See docs for more info."
      )
    endif()
    set(_use_custom_allocator FALSE)
  endif()

  pika_info("Using ${PIKA_WITH_MALLOC} allocator.")

  # Setup Intel amplifier
  if((NOT PIKA_WITH_APEX) AND PIKA_WITH_ITTNOTIFY)

    find_package(Amplifier)
    if(NOT AMPLIFIER_FOUND)
      pika_error(
        "Intel Amplifier could not be found and PIKA_WITH_ITTNOTIFY=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation"
      )
    endif()

    pika_add_config_define(PIKA_HAVE_ITTNOTIFY 1)
    pika_add_config_define(PIKA_HAVE_THREAD_DESCRIPTION)
  endif()

  # convey selected allocator type to the build configuration
  if(NOT PIKA_FIND_PACKAGE)
    pika_add_config_define(PIKA_HAVE_MALLOC "\"${PIKA_WITH_MALLOC}\"")
    if(${PIKA_WITH_MALLOC} STREQUAL "jemalloc")
      if(NOT ("${PIKA_WITH_JEMALLOC_PREFIX}" STREQUAL "<none>")
         AND NOT ("${PIKA_WITH_JEMALLOC_PREFIX}x" STREQUAL "x")
      )
        pika_add_config_define(
          PIKA_HAVE_JEMALLOC_PREFIX ${PIKA_WITH_JEMALLOC_PREFIX}
        )
      endif()
    endif()
  endif()

endif()
