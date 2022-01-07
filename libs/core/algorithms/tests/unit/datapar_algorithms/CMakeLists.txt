# Copyright (c) 2014-2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPXLocal_WITH_DATAPAR_VC OR HPXLocal_WITH_CXX20_EXPERIMENTAL_SIMD)
  set(tests
      ${tests}
      adjacentdifference_datapar
      all_of_datapar
      any_of_datapar
      copy_datapar
      copyn_datapar
      count_datapar
      countif_datapar
      fill_datapar
      filln_datapar
      foreach_datapar
      foreach_datapar_zipiter
      foreachn_datapar
      generate_datapar
      generaten_datapar
      none_of_datapar
      transform_binary_datapar
      transform_binary2_datapar
      transform_reduce_binary_datapar
  )
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Local/Algorithms/Datapar"
  )

  hpx_local_add_unit_test(
    "modules.algorithms.datapar_algorithms" ${test} ${${test}_PARAMETERS}
  )
endforeach()
