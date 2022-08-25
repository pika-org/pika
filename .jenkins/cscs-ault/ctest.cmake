# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2017 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CTEST_TEST_TIMEOUT 300)
set(CTEST_CMAKE_GENERATOR Ninja)
set(CTEST_SITE "cscs(ault)")
set(CTEST_UPDATE_COMMAND "git")
set(CTEST_UPDATE_VERSION_ONLY "ON")
set(CTEST_SUBMIT_RETRY_COUNT 5)
set(CTEST_SUBMIT_RETRY_DELAY 60)

include(ProcessorCount)
processorcount(processor_count)
if(NOT processor_count EQUAL 0)
  set(test_args PARALLEL_LEVEL ${processor_count})
endif()

if("$ENV{ghprbPullId}" STREQUAL "")
  set(bors_branches "staging" "trying")

  # Enable IN_LIST operator
  cmake_policy(SET CMP0057 NEW)
  if("$ENV{git_local_branch}" IN_LIST bors_branches)
    set(CTEST_BUILD_NAME "$ENV{git_local_branch}")

    # Make a string that contains only the PR numbers separated by dashes. The
    # commit messages are assumed to be of the form:
    #
    # "Merge #1 #2 #3"
    # "Try #1 #2 #3:"
    #
    # We strip leading and trailing non-numeric characters, and then replace all
    # intermediate non-numeric characters by a single dash.
    #
    # The result is strings of the form:
    #
    # "1-2-3"
    #
    # which is then added to the CTest build name.
    string(REGEX REPLACE "^[^0-9]+" "" pr_numbers_string "$ENV{git_commit_message}")
    string(REGEX REPLACE "[^0-9]+$" "" pr_numbers_string "${pr_numbers_string}")
    string(REGEX REPLACE "[^0-9]+" "-" pr_numbers_string "${pr_numbers_string}")
    set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}-${pr_numbers_string}")
  else()
    set(CTEST_BUILD_NAME "$ENV{git_local_branch}")
  endif()
  set(CTEST_TRACK "$ENV{git_local_branch}")
else()
  set(CTEST_BUILD_NAME "$ENV{ghprbPullId}")
  set(CTEST_TRACK "Pull_Requests")
endif()

set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}-${CTEST_BUILD_CONFIGURATION_NAME}")

set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} ${CTEST_SOURCE_DIRECTORY}")
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -G${CTEST_CMAKE_GENERATOR}"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -B${CTEST_BINARY_DIRECTORY}"
)
set(CTEST_CONFIGURE_COMMAND
  "${CTEST_CONFIGURE_COMMAND} -DPIKA_WITH_PARALLEL_TESTS_BIND_NONE=ON"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} ${CTEST_CONFIGURE_EXTRA_OPTIONS}"
)

ctest_start(Experimental TRACK "${CTEST_TRACK}")
ctest_update()
ctest_submit(PARTS Update)
ctest_configure()
ctest_submit(PARTS Configure)
ctest_build(TARGET all FLAGS "-k0 ${CTEST_BUILD_EXTRA_OPTIONS}")
ctest_build(TARGET examples FLAGS "-k0 ${CTEST_BUILD_EXTRA_OPTIONS}")
ctest_build(TARGET tests.performance.modules.async_cuda FLAGS "-k0 ${CTEST_BUILD_EXTRA_OPTIONS}")
ctest_build(TARGET tests.regressions.modules.async_cuda FLAGS "-k0 ${CTEST_BUILD_EXTRA_OPTIONS}")
ctest_build(TARGET tests.unit.modules.async_cuda FLAGS "-k0 ${CTEST_BUILD_EXTRA_OPTIONS}")
ctest_submit(PARTS Build)
ctest_test(INCLUDE tests.unit.modules.async_cuda PARALLEL_LEVEL "${CTEST_TEST_PARALLELISM}")
ctest_submit(PARTS Test BUILD_ID CTEST_BUILD_ID)
file(WRITE "${CTEST_JOB_NAME}-cdash-build-id.txt"
     "${CTEST_BUILD_ID}"
)
