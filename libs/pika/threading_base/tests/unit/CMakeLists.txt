# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests resume_suspended_same_thread)

set(resume_suspended_same_thread_PARAMETERS THREADS 2)

if(PIKA_WITH_APEX)
  list(APPEND tests annotation_check)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/ThreadingBase"
  )

  pika_add_unit_test("modules.threading_base" ${test} ${${test}_PARAMETERS})
endforeach()

if(PIKA_WITH_APEX)
  set(launch_types_
      ""
      "Executor "
      "apply "
      "async "
      "deferred "
      "fork "
      "sync "
  )

  # dataflow
  foreach(launch ${launch_types_})
    set(REGEX_MATCH_D_ "${REGEX_MATCH_D_}.*1-${launch}Dataflow")
  endforeach()

  # continuations
  foreach(launch ${launch_types_})
    set(REGEX_MATCH_C_ "${REGEX_MATCH_C_}.*2-${launch}Continuation")
  endforeach()

  # continuations with unwrapping
  foreach(launch ${launch_types_})
    set(REGEX_MATCH_C_ "${REGEX_MATCH_C_}.*3-${launch}Unwrapping Continuation")
  endforeach()

  # annotate_function with std::string and char*
  foreach(type "char" "string")
    set(REGEX_MATCH_ANN_ "${REGEX_MATCH_ANN_}.*4-${type} annotate_function")
  endforeach()

  set_tests_properties(
    tests.unit.modules.threading_base.annotation_check
    PROPERTIES PASS_REGULAR_EXPRESSION
               "${REGEX_MATCH_D_}${REGEX_MATCH_C_}${REGEX_MATCH_ANN_}"
  )
endif()
