# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakeParseArguments)
include(CheckCXXCompilerFlag)

macro(pika_add_link_flag FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args TARGETS CONFIGURATIONS)
  cmake_parse_arguments(
    PIKA_ADD_LINK_FLAG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(PIKA_ADD_LINK_FLAG_PUBLIC)
    set(_dest pika_public_flags)
  else()
    set(_dest pika_private_flags)
  endif()

  set(_targets "none")
  if(PIKA_ADD_LINK_FLAG_TARGETS)
    set(_targets ${PIKA_ADD_LINK_FLAG_TARGETS})
  endif()

  set(_configurations "none")
  if(PIKA_ADD_LINK_FLAG_CONFIGURATIONS)
    set(_configurations "${PIKA_ADD_LINK_FLAG_CONFIGURATIONS}")
  endif()

  foreach(_config ${_configurations})
    foreach(_target ${_targets})
      if(NOT _config STREQUAL "none" AND NOT _target STREQUAL "none")
        set(_flag
            "$<$<AND:$<CONFIG:${_config}>,$<STREQUAL:$<TARGET_PROPERTY:TYPE>,${_target}>:${FLAG}>"
        )
      elseif(_config STREQUAL "none" AND NOT _target STREQUAL "none")
        set(_flag "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,${_target}>:${FLAG}>")
      elseif(NOT _config STREQUAL "none" AND _target STREQUAL "none")
        set(_flag "$<$<CONFIG:${_config}>:${FLAG}>")
      else()
        set(_flag "${FLAG}")
      endif()
      target_link_options(${_dest} INTERFACE "${_flag}")
    endforeach()
  endforeach()
endmacro()

macro(pika_add_link_flag_if_available FLAG)
  set(options PUBLIC)
  set(one_value_args NAME)
  set(multi_value_args TARGETS)
  cmake_parse_arguments(
    PIKA_ADD_LINK_FLAG_IA "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  set(_public)
  if(PIKA_ADD_LINK_FLAG_IA_PUBLIC)
    set(_public PUBLIC)
  endif()

  if(PIKA_ADD_LINK_FLAG_IA_NAME)
    string(TOUPPER ${PIKA_ADD_LINK_FLAG_IA_NAME} _name)
  else()
    string(TOUPPER ${FLAG} _name)
  endif()

  string(REGEX REPLACE " " "" _name ${_name})
  string(REGEX REPLACE "^-+" "" _name ${_name})
  string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
  string(REGEX REPLACE "," "_" _name ${_name})
  string(REGEX REPLACE "\\+" "X" _name ${_name})

  check_cxx_compiler_flag("${FLAG}" WITH_LINKER_FLAG_${_name})
  if(WITH_LINKER_FLAG_${_name})
    pika_add_link_flag(${FLAG} TARGETS ${PIKA_ADD_LINK_FLAG_IA_TARGETS} ${_public})
  else()
    pika_info("Linker \"${FLAG}\" not available.")
  endif()

endmacro()
