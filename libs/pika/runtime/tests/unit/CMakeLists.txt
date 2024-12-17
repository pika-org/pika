# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests configuration_precedence process_mask_flag runtime_initialized)

set(process_mask_flag_PARAMETERS THREADS 2 RUN_SERIAL)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Runtime"
  )

  pika_add_unit_test("modules.runtime" ${test} ${${test}_PARAMETERS})
endforeach()

string(CONCAT no_binding_expected_output "   0: thread binding disabled, on pool \"default\"\n"
              "   1: thread binding disabled, on pool \"default\"\n" "All tests passed."
)

string(
  CONCAT
    compact_binding_expected_output
    "   0: PU L#0\\(P#0\\), Core L#0\\(P#[0-9]+\\)(, Socket L#[0-9]+\\(P#[0-9]+\\))?(, NUMANode L#[0-9]+\\(P#[0-9]+\\))?(, Socket L#[0-9]+\\(P#[0-9]+\\))?, on pool \"default\"\n"
    "   1: PU L#[0-9]+\\(P#1\\), Core L#[0-9]+\\(P#[0-9]+\\)(, Socket L#[0-9]+\\(P#[0-9]+\\))?(, NUMANode L#[0-9]+\\(P#[0-9]+\\))?(, Socket L#[0-9]+\\(P#[0-9]+\\))?, on pool \"default\"\n"
    "All tests passed."
)

string(
  CONCAT
    balanced_binding_expected_output
    "   0: PU L#0\\(P#[0-9]+\\), Core L#0\\(P#0\\)(, Socket L#[0-9]+\\(P#[0-9]+\\))?(, NUMANode L#[0-9]+\\(P#[0-9]+\\))?(, Socket L#[0-9]+\\(P#[0-9]+\\))?, on pool \"default\"\n"
    "   1: PU L#[0-9]+\\(P#[0-9]+\\), Core L#[0-9]+\\(P#1\\)(, Socket L#[0-9]+\\(P#[0-9]+\\))?(, NUMANode L#[0-9]+\\(P#[0-9]+\\))?(, Socket L#[0-9]+\\(P#[0-9]+\\))?, on pool \"default\"\n"
    "All tests passed."
)

if(APPLE)
  set(process_mask_flag_expected_output "${no_binding_expected_output}")
else()
  set(process_mask_flag_expected_output "${compact_binding_expected_output}")
endif()
set_tests_properties(
  tests.unit.modules.runtime.process_mask_flag PROPERTIES PASS_REGULAR_EXPRESSION
                                                          "${process_mask_flag_expected_output}"
)

# Helper function to create tests that check for the correct behaviour of various environment
# variables and command line options, including the precedence of them.
function(
  pika_add_configuration_precedence_test
  base_test_name
  id
  threads
  args
  regex
  env
)
  set(full_name_ ${base_test_name}_${id})
  pika_add_pseudo_target(${full_name_}_test)
  pika_add_pseudo_dependencies(${full_name_}_test ${base_test_name}_test)
  separate_arguments(args)
  pika_add_unit_test(
    "modules.runtime"
    ${full_name_}
    RUN_SERIAL
    THREADS
    ${threads}
    EXECUTABLE
    ${base_test_name}_test
    ARGS
    ${args}
  )
  set_tests_properties(
    tests.unit.modules.runtime.${full_name_} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}"
  )
  if(env)
    separate_arguments(env)
    set_tests_properties(tests.unit.modules.runtime.${full_name_} PROPERTIES ENVIRONMENT ${env})
  endif()
endfunction()

# This tests the default behaviour with two threads. This test is identical to the process_mask_flag
# test above.
if(APPLE)
  set(configuration_precedence_expected_output "${no_binding_expected_output}")
else()
  set(configuration_precedence_expected_output "${balanced_binding_expected_output}")
endif()
pika_add_configuration_precedence_test(
  configuration_precedence 1 2 "--pika:print-bind" "${configuration_precedence_expected_output}" ""
)

# Disable binding through environment variable.
pika_add_configuration_precedence_test(
  configuration_precedence 2 2 "--pika:print-bind" "${no_binding_expected_output}" "PIKA_BIND=none"
)

# Disable binding through through command line option, but enable it through environment variable.
# The former should take precedence.
pika_add_configuration_precedence_test(
  configuration_precedence 3 2 "--pika:print-bind --pika:bind=none" "${no_binding_expected_output}"
  "PIKA_BIND=balanced"
)

# Disable binding through environment variable, but enable it through command line option. The
# latter should take precedence.
pika_add_configuration_precedence_test(
  configuration_precedence 4 2 "--pika:print-bind --pika:bind=balanced"
  "${balanced_binding_expected_output}" "PIKA_BIND=none"
)

string(
  CONCAT
    second_pu_binding_one_thread_expected_output
    "   0: PU L#[0-9]+\\(P#1\\), Core L#[0-9]+\\(P#[0-9]+\\)(, Socket L#[0-9]+\\(P#[0-9]+\\))?(, NUMANode L#[0-9]+\\(P#[0-9]+\\))?(, Socket L#[0-9]+\\(P#[0-9]+\\))?, on pool \"default\"\n"
    "All tests passed."
)
string(CONCAT no_binding_one_thread_expected_output
              "   0: thread binding disabled, on pool \"default\"\n" "All tests passed."
)

# Set process mask through command line option.
if(APPLE)
  set(configuration_precedence_expected_output "${no_binding_one_thread_expected_output}")
else()
  set(configuration_precedence_expected_output "${second_pu_binding_one_thread_expected_output}")
endif()
pika_add_configuration_precedence_test(
  configuration_precedence 5 1 "--pika:print-bind --pika:process-mask=0x2"
  "${configuration_precedence_expected_output}" ""
)

# Set process mask through environment variable.
if(APPLE)
  set(configuration_precedence_expected_output "${no_binding_one_thread_expected_output}")
else()
  set(configuration_precedence_expected_output "${second_pu_binding_one_thread_expected_output}")
endif()
pika_add_configuration_precedence_test(
  configuration_precedence 6 1 "--pika:print-bind" "${configuration_precedence_expected_output}"
  "PIKA_PROCESS_MASK=0x2"
)

# Set process mask through environment variable and command line option. The latter should take
# precedence.
if(APPLE)
  set(configuration_precedence_expected_output "${no_binding_one_thread_expected_output}")
else()
  set(configuration_precedence_expected_output "${second_pu_binding_one_thread_expected_output}")
endif()
pika_add_configuration_precedence_test(
  configuration_precedence 7 1 "--pika:print-bind --pika:process-mask=0x2"
  "${configuration_precedence_expected_output}" "PIKA_PROCESS_MASK=0x4"
)

string(CONCAT one_thread_expected_output "   0:.*\n" "All tests passed.")
string(CONCAT two_threads_expected_output "   0:.*\n" "   1:.*\n" "All tests passed.")

# Set number of threads through command line option.
pika_add_configuration_precedence_test(
  configuration_precedence 8 -3 "--pika:print-bind --pika:threads=1"
  "${one_thread_expected_output}" ""
)

# Set number of threads through environment variable.
pika_add_configuration_precedence_test(
  configuration_precedence 9 -3 "--pika:print-bind" "${two_threads_expected_output}"
  "PIKA_THREADS=2"
)

# Set number of threads through environment variable and command line option.
pika_add_configuration_precedence_test(
  configuration_precedence 10 -3 "--pika:print-bind --pika:threads=2"
  "${two_threads_expected_output}" "PIKA_THREADS=1"
)
