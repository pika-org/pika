# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests schedule_last)

# ##################################################################################################
foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} ${${test}_LIBRARIES}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Schedulers"
  )

  pika_add_unit_test("modules.schedulers" ${test} ${${test}_PARAMETERS})
endforeach()
