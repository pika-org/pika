# Copyright (c) 2014-2017 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(boost_library_dependencies ${Boost_LIBRARIES})

set(benchmarks
    async_overheads
    coroutines_call_overhead
    delay_baseline
    delay_baseline_threaded
    function_object_wrapper_overhead
    future_overhead_local
    future_overhead_local_report
    heterogeneous_timed_task_spawn
    tls_overhead
    native_tls_overhead
    parent_vs_child_stealing
    print_heterogeneous_payloads
    resume_suspend
    skynet
    stream_local
    stream_local_report
    wait_all_timings
)

if(NOT PIKA_WITH_SANITIZERS)
  list(APPEND benchmarks start_stop)
endif()

if(PIKA_WITH_LIBCDS)
  list(APPEND benchmarks libcds_hazard_pointer_overhead)
endif()

if(PIKA_WITH_CXX20_EXPERIMENTAL_SIMD OR PIKA_WITH_DATAPAR_VC)
  list(APPEND benchmarks transform_reduce_binary_scaling)
endif()

if(PIKA_WITH_EXAMPLES_OPENMP)
  list(APPEND benchmarks openmp_homogeneous_timed_task_spawn
       openmp_parallel_region
  )

  set(openmp_homogeneous_timed_task_spawn_FLAGS
      NOLIBS DEPENDENCIES ${boost_library_dependencies} pika
  )
  set(openmp_parallel_region_FLAGS NOLIBS DEPENDENCIES pika)
endif()

if(PIKA_WITH_EXAMPLES_QTHREADS)
  list(APPEND benchmarks qthreads_homogeneous_timed_task_spawn
       qthreads_heterogeneous_timed_task_spawn
  )

  set(qthreads_homogeneous_timed_task_spawn_FLAGS
      NOLIBS DEPENDENCIES ${boost_library_dependencies} ${QTHREADS_LIBRARIES}
      pika
  )

  set(qthreads_heterogeneous_timed_task_spawn_FLAGS
      NOLIBS DEPENDENCIES ${boost_library_dependencies} ${QTHREADS_LIBRARIES}
      pika
  )

  set(qthreads_heterogeneous_timed_task_spawn_INCLUDE_DIRECTORIES
      ${QTHREADS_INCLUDE_DIR}
  )
endif()

if(PIKA_WITH_EXAMPLES_TBB)
  list(APPEND benchmarks tbb_homogeneous_timed_task_spawn)

  set(tbb_homogeneous_timed_task_spawn_FLAGS
      NOLIBS DEPENDENCIES ${boost_library_dependencies} ${TBB_LIBRARIES}
      pika
  )

  set(tbb_homogeneous_timed_task_spawn_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIR})
endif()

set(delay_baseline_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies}
                         pika
)
set(delay_baseline_threaded_FLAGS DEPENDENCIES ${boost_library_dependencies}
                                  pika
)
set(print_heterogeneous_payloads_FLAGS NOLIBS DEPENDENCIES
                                       ${boost_library_dependencies} pika
)
set(resume_suspend_FLAGS DEPENDENCIES pika_timing)
set(native_tls_overhead_LIBRARIES pika_dependencies_boost)

set(future_overhead_local_PARAMETERS THREADS_PER_LOCALITY 4)
set(future_overhead_local_report_PARAMETERS THREADS_PER_LOCALITY 4)

# These tests do not run on pika threads, so we don't want to pass pika params
# into them
set(delay_baseline_PARAMETERS NO_PIKA_MAIN)
set(delay_baseline_threaded_PARAMETERS NO_PIKA_MAIN)
set(function_object_wrapper_overhead_PARAMETERS NO_PIKA_MAIN)
set(nonconcurrent_fifo_overhead_PARAMETERS NO_PIKA_MAIN)
set(nonconcurrent_lifo_overhead_PARAMETERS NO_PIKA_MAIN)
set(print_heterogeneous_payloads_PARAMETERS NO_PIKA_MAIN)

# These tests fail, so I am marking them as non pika tests until they are fixed
set(print_heterogeneous_payloads_PARAMETERS NO_PIKA_MAIN)
set(skynet_PARAMETERS NO_PIKA_MAIN)
set(timed_task_spawn_PARAMETERS NO_PIKA_MAIN)
set(pika_tls_overhead_PARAMETERS NO_PIKA_MAIN)
set(native_tls_overhead_PARAMETERS NO_PIKA_MAIN)
set(coroutines_call_overhead_PARAMETERS NO_PIKA_MAIN)

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp ${${benchmark}_SOURCES})

  source_group("Source Files" FILES ${sources})
  if(${benchmark}_HEADERS)
    source_group("Header Files" FILES ${${benchmark}_HEADERS})
    set(${benchmark}_HEADERS HEADERS ${${benchmark}_HEADERS})
  endif()

  # add example executable
  pika_add_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS} ${${benchmark}_HEADERS}
    EXCLUDE_FROM_ALL
    FOLDER "Benchmarks/Local"
  )

  target_include_directories(
    ${benchmark}_test SYSTEM PRIVATE ${${benchmark}_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(${benchmark}_test PRIVATE ${${benchmark}_LIBRARIES})

  if(NOT "${${benchmark}_PARAMETERS}" MATCHES NO_PIKA_MAIN)
    pika_add_performance_test(
      "local" ${benchmark} ${${benchmark}_PARAMETERS}
    )
  endif()

endforeach()

if(PIKA_WITH_LIBCDS)
  target_link_libraries(libcds_hazard_pointer_overhead_test PRIVATE cds)
endif()

if(PIKA_WITH_EXAMPLES_OPENMP)
  set_target_properties(
    openmp_homogeneous_timed_task_spawn_test PROPERTIES COMPILE_FLAGS
                                                        ${OpenMP_CXX_FLAGS}
  )
  set_target_properties(
    openmp_homogeneous_timed_task_spawn_test PROPERTIES LINK_FLAGS
                                                        ${OpenMP_CXX_FLAGS}
  )
  set_target_properties(
    openmp_parallel_region_test PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
  )
  set_target_properties(
    openmp_parallel_region_test PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS}
  )
endif()

pika_add_pseudo_target(tests.performance.local.htts_v2)
add_subdirectory(htts_v2)
pika_add_pseudo_dependencies(
  tests.performance.local tests.performance.local.htts_v2
)
