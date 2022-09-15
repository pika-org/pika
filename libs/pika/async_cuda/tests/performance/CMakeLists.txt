# Copyright (c) 2020 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks cuda_scheduler_throughput)

if(NOT PIKA_WITH_CLANG_CUDA)
  list(APPEND benchmarks synchronize)
endif()

set(synchronize_GPU ON)
set(cuda_scheduler_throughput_PARAMETERS THREADS 1)

foreach(benchmark ${benchmarks})

  if(${${benchmark}_GPU})
    set(sources ${benchmark}.cu)
  else()
    set(sources ${benchmark}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  pika_add_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    DEPENDENCIES pika_async_cuda ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/AsyncCuda"
  )

  # add a custom target for this benchmark
  pika_add_performance_test(
    "modules.async_cuda" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
