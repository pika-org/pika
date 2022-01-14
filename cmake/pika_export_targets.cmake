# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(pika_export_targets)
  foreach(target ${ARGN})
    list(FIND PIKA_EXPORT_TARGETS ${target} _found)
    if(_found EQUAL -1)
      set(PIKA_EXPORT_TARGETS
          ${PIKA_EXPORT_TARGETS} ${target}
          CACHE INTERNAL "" FORCE
      )
    endif()
  endforeach()
endfunction(pika_export_targets)

function(pika_export_internal_targets)
  foreach(target ${ARGN})
    list(FIND PIKA_EXPORT_INTERNAL_TARGETS ${target} _found)
    if(_found EQUAL -1)
      set(PIKA_EXPORT_INTERNAL_TARGETS
          ${PIKA_EXPORT_INTERNAL_TARGETS} ${target}
          CACHE INTERNAL "" FORCE
      )
    endif()
  endforeach()
endfunction(pika_export_internal_targets)
