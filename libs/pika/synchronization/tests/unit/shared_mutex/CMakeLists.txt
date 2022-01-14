# Copyright (c) 2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests shared_mutex1 shared_mutex2)

set(shared_mutex1_PARAMETERS THREADS_PER_LOCALITY 4)
set(shared_mutex2_PARAMETERS THREADS_PER_LOCALITY 4)

set(shared_mutex1_FLAGS DEPENDENCIES PRIVATE pika_dependencies_boost)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Local/Synchronization/SharedMutex"
  )

  pika_add_unit_test(
    "modules.synchronization.shared_mutex" ${test} ${${test}_PARAMETERS}
  )

endforeach()
