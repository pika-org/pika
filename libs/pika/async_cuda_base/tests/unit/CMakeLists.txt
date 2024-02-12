# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests cublas_handle cuda_stream cusolver_handle)

set(cublas_handle_GPU ON)
set(cuda_stream_GPU ON)
set(cusolver_handle_GPU ON)

foreach(test ${tests})
  if(${${test}_GPU})
    set(sources ${test}.cu)
  else()
    set(sources ${test}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  # add test executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    DEPENDENCIES ${${test}_DEPENDENCIES}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/AsyncCudaBase"
  )

  pika_add_unit_test("modules.async_cuda_base" ${test} ${${test}_PARAMETERS} RUN_SERIAL)
endforeach()
