# Copyright (c) 2015      John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# -------------------------------------------------------------------------------
# adds pika_ prefix to give pika_${name} to libraries and components
# -------------------------------------------------------------------------------
function(pika_set_lib_name target name)
  # there is no need to change debug/release names explicitly as we use CMAKE_DEBUG_POSTFIX to alter
  # debug names

  pika_debug("pika_set_lib_name: target:" ${target} "name: " ${name})
  set_target_properties(
    ${target}
    PROPERTIES OUTPUT_NAME pika_${name}
               DEBUG_OUTPUT_NAME pika_${name}
               RELEASE_OUTPUT_NAME pika_${name}
               MINSIZEREL_OUTPUT_NAME pika_${name}
               RELWITHDEBINFO_OUTPUT_NAME pika_${name}
  )
endfunction()
