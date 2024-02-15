# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakeDependentOption)
include(CMakeParseArguments)

set(PIKA_OPTION_CATEGORIES "Generic" "Build Targets" "Thread Manager" "Profiling" "Debugging")

macro(pika_option option type description default)
  set(options ADVANCED)
  set(one_value_args CATEGORY DEPENDS)
  set(multi_value_args STRINGS)
  cmake_parse_arguments(PIKA_OPTION "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if("${type}" STREQUAL "BOOL")
    # Use regular CMake options for booleans
    if(NOT PIKA_OPTION_DEPENDS)
      option("${option}" "${description}" "${default}")
    else()
      cmake_dependent_option("${option}" "${description}" "${default}" "${PIKA_OPTION_DEPENDS}" OFF)
    endif()
  else()
    if(PIKA_OPTION_DEPENDS)
      message(FATAL_ERROR "pika_option DEPENDS keyword can only be used with BOOL options")
    endif()
    # Use custom cache variables for other types
    if(NOT DEFINED ${option})
      set(${option}
          ${default}
          CACHE ${type} "${description}" FORCE
      )
    else()
      get_property(
        _option_is_cache_property
        CACHE "${option}"
        PROPERTY TYPE
        SET
      )
      if(NOT _option_is_cache_property)
        set(${option}
            ${default}
            CACHE ${type} "${description}" FORCE
        )
        if(PIKA_OPTION_ADVANCED)
          mark_as_advanced(${option})
        endif()
      else()
        set_property(CACHE "${option}" PROPERTY HELPSTRING "${description}")
        set_property(CACHE "${option}" PROPERTY TYPE "${type}")
      endif()
    endif()

    if(PIKA_OPTION_STRINGS)
      if("${type}" STREQUAL "STRING")
        set_property(CACHE "${option}" PROPERTY STRINGS "${PIKA_OPTION_STRINGS}")
      else()
        message(FATAL_ERROR "pika_option(): STRINGS can only be used if type is STRING !")
      endif()
    endif()
  endif()

  if(PIKA_OPTION_ADVANCED)
    mark_as_advanced(${option})
  endif()

  set_property(GLOBAL APPEND PROPERTY PIKA_MODULE_CONFIG_PIKA ${option})

  set(_category "Generic")
  if(PIKA_OPTION_CATEGORY)
    set(_category "${PIKA_OPTION_CATEGORY}")
  endif()
  set(${option}Category
      ${_category}
      CACHE INTERNAL ""
  )
endmacro()
