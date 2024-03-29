# Copyright (c) 2011-2013 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(jacobi_smp_applications jacobi_pika)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  list(APPEND jacobi_smp_applications jacobi_nonuniform_pika)
endif()
set(jacobi_pika_PARAMETERS THREADS 4)
set(jacobi_nonuniform_pika_PARAMETERS THREADS 4)

set(jacobi_pika_sources jacobi.cpp)
set(jacobi_nonuniform_pika_sources jacobi_nonuniform.cpp)

set(disabled_tests # Disabled because requires external input data. TODO: Download data when
    # running test.
    jacobi_nonuniform_pika
)

if(PIKA_WITH_EXAMPLES_OPENMP)
  set(jacobi_smp_applications ${jacobi_smp_applications} jacobi_omp_static jacobi_omp_dynamic
                              jacobi_nonuniform_omp_static jacobi_nonuniform_omp_dynamic
  )
  set(jacobi_omp_static_sources jacobi.cpp)
  set(jacobi_omp_dynamic_sources jacobi.cpp)
  set(jacobi_nonuniform_omp_static_sources jacobi_nonuniform.cpp)
  set(jacobi_nonuniform_omp_dynamic_sources jacobi_nonuniform.cpp)

  set(disabled_tests ${disabled_tests} jacobi_nonuniform_omp_static jacobi_nonuniform_omp_dynamic)
endif()

set(jacobi_pika_sources jacobi.cpp)
set(jacobi_nonuniform_pika_sources jacobi_nonuniform.cpp)

foreach(jacobi_smp_application ${jacobi_smp_applications})
  set(sources ${${jacobi_smp_application}_sources} ${jacobi_smp_application}.cpp)

  source_group("Source Files" FILES ${sources})

  # add jacobi_smp executable
  pika_add_executable(
    ${jacobi_smp_application} INTERNAL_FLAGS MODULE jacobi_smp
    SOURCES ${sources} ${${jacobi_smp_application}_FLAGS}
    FOLDER "Examples/JacobiSMP"
  )

  pika_add_example_target_dependencies("jacobi_smp" ${jacobi_smp_application})

  if(PIKA_WITH_TESTS
     AND PIKA_WITH_TESTS_EXAMPLES
     AND NOT ("${jacobi_smp_application}" IN_LIST disabled_tests)
  )
    pika_add_example_test(
      "jacobi_smp" ${jacobi_smp_application} ${${jacobi_smp_application}_PARAMETERS}
    )
  endif()
endforeach()

if(PIKA_WITH_EXAMPLES_OPENMP)
  set_target_properties(jacobi_omp_static PROPERTIES COMPILE_DEFINITIONS JACOBI_SMP_NO_pika)
  set_target_properties(
    jacobi_omp_static PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(jacobi_omp_static PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

  set_target_properties(jacobi_omp_dynamic PROPERTIES COMPILE_DEFINITIONS JACOBI_SMP_NO_pika)
  set_target_properties(
    jacobi_omp_dynamic PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(jacobi_omp_dynamic PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

  set_target_properties(
    jacobi_nonuniform_omp_static PROPERTIES COMPILE_DEFINITIONS JACOBI_SMP_NO_pika
  )
  set_target_properties(
    jacobi_nonuniform_omp_static PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(jacobi_nonuniform_omp_static PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

  set_target_properties(
    jacobi_nonuniform_omp_dynamic PROPERTIES COMPILE_DEFINITIONS JACOBI_SMP_NO_pika
  )
  set_target_properties(
    jacobi_nonuniform_omp_dynamic PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(jacobi_nonuniform_omp_dynamic PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
endif()
