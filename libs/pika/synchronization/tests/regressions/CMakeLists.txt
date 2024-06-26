# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests ignore_while_locked_1485)

set(ignore_while_locked_1485_PARAMETERS THREADS 2)

# Create test cases
foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/Synchronization"
  )

  target_link_libraries(${test}_test PRIVATE ${${test}_LIBRARIES})
  pika_add_regression_test("modules.synchronization" ${test} ${${test}_PARAMETERS})

endforeach()
