# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests resume_suspended_same_thread)

set(resume_suspended_same_thread_PARAMETERS THREADS 2)

if(PIKA_WITH_APEX)
  list(APPEND tests annotation_check_senders)
  set(annotation_check_senders_PARAMETERS THREADS 2 WRAPPER
                                          "${PROJECT_SOURCE_DIR}/tools/sort-output.sh"
  )
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
  string(
    CONCAT REGEX_MATCH_S_
           "0-execute-no-parent-A :[ ]+1[ ]+.*"
           "0-execute-no-parent-C :[ ]+1[ ]+.*"
           "1-execute-parent :[ ]+2[ ]+.*"
           "1-execute-parent-A :[ ]+1[ ]+.*"
           "1-execute-parent-C :[ ]+1[ ]+.*"
           "2-schedule-no-parent-A :[ ]+1[ ]+.*"
           "2-schedule-no-parent-B :[ ]+1[ ]+.*"
           "2-schedule-no-parent-C :[ ]+1[ ]+.*"
           "2-schedule-no-parent-D :[ ]+1[ ]+.*"
           "2-schedule-no-parent-E :[ ]+1[ ]+.*"
           "2-schedule-no-parent-F :[ ]+1[ ]+.*"
           "2-schedule-no-parent-G :[ ]+2[ ]+.*"
           "2-schedule-no-parent-H :[ ]+1[ ]+.*"
           "3-schedule-parent :[ ]+11[ ]+.*"
           "3-schedule-parent-A :[ ]+1[ ]+.*"
           "3-schedule-parent-B :[ ]+1[ ]+.*"
           "3-schedule-parent-C :[ ]+1[ ]+.*"
           "3-schedule-parent-D :[ ]+1[ ]+.*"
           "3-schedule-parent-E :[ ]+1[ ]+.*"
           "3-schedule-parent-F :[ ]+1[ ]+.*"
           "3-schedule-parent-G :[ ]+2[ ]+.*"
           "3-schedule-parent-H :[ ]+1[ ]+.*"
           "<unknown> :[ ]+12[ ]+.*"
  )

  set_tests_properties(
    tests.unit.modules.threading_base.annotation_check_senders PROPERTIES PASS_REGULAR_EXPRESSION
                                                                          "${REGEX_MATCH_S_}"
  )
endif()
