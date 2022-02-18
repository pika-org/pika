# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(pika_add_test category name)
  set(options FAILURE_EXPECTED RUN_SERIAL)
  set(one_value_args EXECUTABLE LOCALITIES THREADS TIMEOUT RUNWRAPPER)
  set(multi_value_args ARGS)
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT ${name}_LOCALITIES)
    set(${name}_LOCALITIES 1)
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
  elseif(TARGET ${${name}_EXECUTABLE})
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}>")
  else()
    set(_exe "${${name}_EXECUTABLE}")
  endif()

  if(${name}_RUN_SERIAL)
    set(run_serial TRUE)
  endif()

  # If --pika:threads=cores or all
  if(${name}_THREADS LESS_EQUAL 0)
    set(run_serial TRUE)
    if(${name}_THREADS EQUAL -1)
      set(${name}_THREADS "all")
    elseif(${name}_THREADS EQUAL -2)
      set(${name}_THREADS "cores")
    endif()
  endif()

  set(args "--pika:threads=${${name}_THREADS}")
  if(${PIKA_WITH_TESTS_DEBUG_LOG})
    set(args ${args}
             "--pika:debug-pika-log=${PIKA_WITH_TESTS_DEBUG_LOG_DESTINATION}"
    )
  endif()

  if(${PIKA_WITH_PARALLEL_TESTS_BIND_NONE}
     AND NOT run_serial
     AND NOT "${name}_RUNWRAPPER"
  )
    set(args ${args} "--pika:bind=none")
  endif()

  set(args "${${name}_UNPARSED_ARGUMENTS}" ${args})

  set(_script_location ${PROJECT_BINARY_DIR})

  set(cmd ${_exe})

  if(${name}_RUNWRAPPER)
    list(PREPEND cmd "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}"
         "${${name}_LOCALITIES}"
    )
  endif()

  set(_full_name "${category}.${name}")
  add_test(NAME "${category}.${name}" COMMAND ${cmd} ${args})
  if(${run_serial})
    set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
  endif()
  if(${name}_TIMEOUT)
    set_tests_properties("${_full_name}" PROPERTIES TIMEOUT ${${name}_TIMEOUT})
  endif()
  if(${name}_FAILURE_EXPECTED)
    set_tests_properties("${_full_name}" PROPERTIES WILL_FAIL TRUE)
  endif()

  # Only real tests, i.e. executables ending in _test, link to pika_testing
  if(TARGET ${${name}_EXECUTABLE}_test)
    target_link_libraries(${${name}_EXECUTABLE}_test PRIVATE pika_testing)
  endif()

endfunction(pika_add_test)

function(pika_add_test_target_dependencies category name)
  set(one_value_args PSEUDO_DEPS_NAME)
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
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
    pika_add_pseudo_dependencies(
      ${category}.${name} ${${name}_PSEUDO_DEPS_NAME}${_ext}
    )
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
    pika_add_test_target_dependencies(
      tests.${category}.${subcategory} ${name} ${ARGN}
    )
  endif()
endfunction(pika_add_test_and_deps_test)

function(pika_add_unit_test subcategory name)
  pika_add_test_and_deps_test("unit" "${subcategory}" ${name} ${ARGN})
endfunction(pika_add_unit_test)

function(pika_add_regression_test subcategory name)
  # ARGN needed in case we add a test with the same executable
  pika_add_test_and_deps_test("regressions" "${subcategory}" ${name} ${ARGN})
endfunction(pika_add_regression_test)

function(pika_add_performance_test subcategory name)
  pika_add_test_and_deps_test(
    "performance" "${subcategory}" ${name} ${ARGN} RUN_SERIAL
  )
endfunction(pika_add_performance_test)

function(pika_add_example_test subcategory name)
  pika_add_test_and_deps_test("examples" "${subcategory}" ${name} ${ARGN})
endfunction(pika_add_example_test)

# To create target examples.<name> when calling make examples need 2 distinct
# rules for examples and tests.examples
function(pika_add_example_target_dependencies subcategory name)
  set(options DEPS_ONLY)
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
  if(NOT ${name}_DEPS_ONLY)
    # Add a custom target for this example
    pika_add_pseudo_target(examples.${subcategory}.${name})
  endif()
  # Make pseudo-targets depend on master pseudo-target
  pika_add_pseudo_dependencies(
    examples.${subcategory} examples.${subcategory}.${name}
  )
  # Add dependencies to pseudo-target
  pika_add_pseudo_dependencies(examples.${subcategory}.${name} ${name})
endfunction(pika_add_example_target_dependencies)
