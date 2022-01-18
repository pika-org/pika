# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(pika_to_string var)
  set(_var "")

  foreach(_arg ${ARGN})
    string(REPLACE "\\" "/" _arg ${_arg})
    if("${_var}" STREQUAL "")
      set(_var "${_arg}")
    else()
      set(_var "${_var} ${_arg}")
    endif()
  endforeach()

  set(${var}
      ${_var}
      PARENT_SCOPE
  )
endfunction()

function(pika_info)
  set(msg)
  pika_to_string(msg ${ARGN})
  message(STATUS "${msg}")
  unset(args)
endfunction()

function(pika_debug)
  if("${PIKA_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(msg "DEBUG:")
    pika_to_string(msg ${ARGN})
    message(STATUS "${msg}")
  endif()
endfunction()

function(pika_warn)
  if("${PIKA_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    set(msg "WARNING:")
    pika_to_string(msg ${ARGN})
    message(STATUS "${msg}")
  endif()
endfunction()

function(pika_error)
  set(msg "ERROR:")
  pika_to_string(msg ${ARGN})
  message(FATAL_ERROR "${msg}")
endfunction()

function(pika_message level)
  if("${level}" MATCHES "ERROR|error|Error")
    pika_error(${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    pika_warn(${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    pika_debug(${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    pika_info(${ARGN})
  else()
    pika_error(
      "message" "\"${level}\" is not an pika configuration logging level."
    )
  endif()
endfunction()

function(pika_config_loglevel level return)
  set(${return}
      FALSE
      PARENT_SCOPE
  )
  if("${PIKA_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error"
     AND "${level}" MATCHES "ERROR|error|Error"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${PIKA_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn"
         AND "${level}" MATCHES "WARN|warn|Warn"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${PIKA_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug"
         AND "${level}" MATCHES "DEBUG|debug|Debug"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  elseif("${PIKA_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info"
         AND "${level}" MATCHES "INFO|info|Info"
  )
    set(${return}
        TRUE
        PARENT_SCOPE
    )
  endif()
endfunction()

function(pika_print_list level message list)
  pika_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      pika_message(${level} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      pika_message(${level} "${message} is empty.")
    endif()
  endif()
endfunction()
