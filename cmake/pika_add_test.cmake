# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_TESTS_VALGRIND)
  find_program(VALGRIND_EXECUTABLE valgrind REQUIRED)
endif()

function(pika_add_test category name)
  set(options FAILURE_EXPECTED RUN_SERIAL TESTING PERFORMANCE_TESTING MPIWRAPPER)
  set(one_value_args COST EXECUTABLE RANKS THREADS TIMEOUT WRAPPER)
  set(multi_value_args ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_RANKS)
    set(${name}_RANKS 1)
  endif()

  if(NOT ${name}_THREADS)
    set(${name}_THREADS 1)
  elseif(PIKA_WITH_TESTS_MAX_THREADS GREATER 0 AND ${name}_THREADS GREATER
                                                   PIKA_WITH_TESTS_MAX_THREADS
  )
    set(${name}_THREADS ${PIKA_WITH_TESTS_MAX_THREADS})
  endif()

  if(NOT ${name}_EXECUTABLE)
    set(${name}_EXECUTABLE ${name})
  endif()

  if(TARGET ${${name}_EXECUTABLE}_test)
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}_test>")
    set(target ${${name}_EXECUTABLE}_test)
  elseif(TARGET ${${name}_EXECUTABLE})
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}>")
    set(target ${${name}_EXECUTABLE})
  else()
    set(_exe "${${name}_EXECUTABLE}")
    set(target "")
  endif()

  if(${name}_RUN_SERIAL)
    set(run_serial TRUE)
  endif()

  # If --pika:threads=cores or all
  if(${name}_THREADS LESS_EQUAL 0)
    set(run_serial TRUE)
    if(${name}_THREADS EQUAL -1)
      set(${name}_THREADS "all")
      set(run_serial TRUE)
    elseif(${name}_THREADS EQUAL -2)
      set(${name}_THREADS "cores")
      set(run_serial TRUE)
    endif()
  endif()

  set(args "--pika:threads=${${name}_THREADS}")
  if(PIKA_WITH_TESTS_DEBUG_LOG)
    set(args ${args} "--pika:log-destination=${PIKA_WITH_TESTS_DEBUG_LOG_DESTINATION}")
    set(args ${args} "--pika:log-level=0")
  endif()

  if(PIKA_WITH_PARALLEL_TESTS_BIND_NONE
     AND NOT run_serial
     AND NOT "${name}_MPIWRAPPER"
  )
    set(args ${args} "--pika:bind=none")
  endif()

  set(args "${${name}_ARGS}" "${${name}_UNPARSED_ARGUMENTS}" ${args})

  set(_script_location ${PROJECT_BINARY_DIR})

  set(cmd ${_exe})

  if(${name}_WRAPPER)
    list(PREPEND cmd "${${name}_WRAPPER}" ${_preflags_list_})
  endif()

  if(${name}_MPIWRAPPER)
    set(_preflags_list_ ${MPIEXEC_PREFLAGS})
    separate_arguments(_preflags_list_)
    list(PREPEND cmd "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${${name}_RANKS}"
         ${_preflags_list_}
    )
  endif()

  if(PIKA_WITH_TESTS_VALGRIND)
    set(valgrind_cmd ${VALGRIND_EXECUTABLE} ${PIKA_WITH_TESTS_VALGRIND_OPTIONS})
  endif()

  set(_full_name "${category}.${name}")
  add_test(NAME "${category}.${name}" COMMAND ${valgrind_cmd} ${cmd} ${args})
  if(${run_serial})
    set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
  endif()
  if(${name}_COST)
    set_tests_properties("${_full_name}" PROPERTIES COST ${${name}_COST})
  endif()
  if(${name}_TIMEOUT)
    set_tests_properties("${_full_name}" PROPERTIES TIMEOUT ${${name}_TIMEOUT})
  endif()
  if(${name}_FAILURE_EXPECTED)
    set_tests_properties("${_full_name}" PROPERTIES WILL_FAIL TRUE)
  endif()

  if(NOT "${target}" STREQUAL "" AND ${name}_TESTING)
    target_link_libraries(${target} PRIVATE pika_testing)
  endif()

  if(NOT "${target}" STREQUAL "" AND ${name}_PERFORMANCE_TESTING)
    target_link_libraries(${target} PRIVATE pika_performance_testing)
  endif()

endfunction(pika_add_test)

function(pika_add_test_target_dependencies category name)
  set(one_value_args PSEUDO_DEPS_NAME)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  # default target_extension is _test but for examples.* target, it may vary
  if(NOT ("${category}" MATCHES "tests.examples*"))
    set(_ext "_test")
  endif()
  # Add a custom target for this example
  pika_add_pseudo_target(${category}.${name})
  # Make pseudo-targets depend on master pseudo-target
  pika_add_pseudo_dependencies(${category} ${category}.${name})
  # Add dependencies to pseudo-target
  if(${name}_PSEUDO_DEPS_NAME)
    # When the test depend on another executable name
    pika_add_pseudo_dependencies(${category}.${name} ${${name}_PSEUDO_DEPS_NAME}${_ext})
  else()
    pika_add_pseudo_dependencies(${category}.${name} ${name}${_ext})
  endif()
endfunction(pika_add_test_target_dependencies)

# To add test to the category root as in tests/regressions/ with correct name
function(pika_add_test_and_deps_test category subcategory name)
  if("${subcategory}" STREQUAL "")
    pika_add_test(tests.${category} ${name} ${ARGN})
    pika_add_test_target_dependencies(tests.${category} ${name} ${ARGN})
  else()
    pika_add_test(tests.${category}.${subcategory} ${name} ${ARGN})
    pika_add_test_target_dependencies(tests.${category}.${subcategory} ${name} ${ARGN})
  endif()
endfunction(pika_add_test_and_deps_test)

# Only unit and regression tests link to the testing library. Performance tests and examples don't
# link to the testing library. Performance tests link to the performance_testing library.
function(pika_add_unit_test subcategory name)
  pika_add_test_and_deps_test("unit" "${subcategory}" ${name} ${ARGN} TESTING)
endfunction(pika_add_unit_test)

function(pika_add_regression_test subcategory name)
  # ARGN needed in case we add a test with the same executable
  pika_add_test_and_deps_test("regressions" "${subcategory}" ${name} ${ARGN} TESTING)
endfunction(pika_add_regression_test)

function(pika_add_performance_test subcategory name)
  pika_add_test_and_deps_test(
    "performance" "${subcategory}" ${name} ${ARGN} RUN_SERIAL PERFORMANCE_TESTING
  )
endfunction(pika_add_performance_test)

function(pika_add_example_test subcategory name)
  pika_add_test_and_deps_test("examples" "${subcategory}" ${name} ${ARGN})
endfunction(pika_add_example_test)

# To create target examples.<name> when calling make examples need 2 distinct rules for examples and
# tests.examples
function(pika_add_example_target_dependencies subcategory name)
  set(options DEPS_ONLY)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT ${name}_DEPS_ONLY)
    # Add a custom target for this example
    pika_add_pseudo_target(examples.${subcategory}.${name})
  endif()
  # Make pseudo-targets depend on master pseudo-target
  pika_add_pseudo_dependencies(examples.${subcategory} examples.${subcategory}.${name})
  # Add dependencies to pseudo-target
  pika_add_pseudo_dependencies(examples.${subcategory}.${name} ${name})
endfunction(pika_add_example_target_dependencies)
