# Copyright (c) 2015 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(pika_add_compile_test category name)
  set(options FAILURE_EXPECTED NOLIBS OBJECT)
  set(one_value_args SOURCE_ROOT FOLDER)
  set(multi_value_args SOURCES DEPENDENCIES)

  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  set(_additional_flags)
  if(${name}_NOLIBS)
    set(_additional_flags ${_additional_flags} NOLIBS)
  endif()
  if(${name}_OBJECT)
    set(_additional_flags ${_additional_flags} OBJECT)
  endif()

  string(REGEX REPLACE "\\." "_" test_name "${category}.${name}")

  if(${name}_OBJECT)
    pika_add_library(
      ${test_name}
      SOURCE_ROOT ${${name}_SOURCE_ROOT}
      SOURCES ${${name}_SOURCES}
      EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD
      FOLDER ${${name}_FOLDER}
      DEPENDENCIES ${${name}_DEPENDENCIES} ${_additional_flags}
    )
  else()
    pika_add_executable(
      ${test_name}
      SOURCE_ROOT ${${name}_SOURCE_ROOT}
      SOURCES ${${name}_SOURCES}
      EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD
      FOLDER ${${name}_FOLDER}
      DEPENDENCIES ${${name}_DEPENDENCIES} ${_additional_flags}
    )
  endif()

  add_test(
    NAME "${category}.${name}"
    COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target ${test_name}
            --config $<CONFIGURATION>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )

  if(${name}_FAILURE_EXPECTED)
    set_tests_properties("${category}.${name}" PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction(pika_add_compile_test)

function(pika_add_compile_test_target_dependencies category name)
  pika_add_pseudo_target(${category}.${name})
  # make pseudo-targets depend on master pseudo-target
  pika_add_pseudo_dependencies(${category} ${category}.${name})
endfunction(pika_add_compile_test_target_dependencies)

# To add test to the category root as in tests/regressions/ with correct name
function(pika_add_test_and_deps_compile_test category subcategory name)
  if("${subcategory}" STREQUAL "")
    pika_add_compile_test(tests.${category} ${name} ${ARGN})
    pika_add_compile_test_target_dependencies(tests.${category} ${name})
  else()
    pika_add_compile_test(tests.${category}.${subcategory} ${name} ${ARGN})
    pika_add_compile_test_target_dependencies(
      tests.${category}.${subcategory} ${name}
    )
  endif()
endfunction(pika_add_test_and_deps_compile_test)

function(pika_add_unit_compile_test subcategory name)
  pika_add_test_and_deps_compile_test(
    "unit" "${subcategory}" ${name} ${ARGN}
  )
endfunction(pika_add_unit_compile_test)

function(pika_add_regression_compile_test subcategory name)
  pika_add_test_and_deps_compile_test(
    "regressions" "${subcategory}" ${name} ${ARGN}
  )
endfunction(pika_add_regression_compile_test)

function(pika_add_headers_compile_test subcategory name)
  # Important to keep the double quotes around subcategory otherwise don't
  # consider empty argument but just remove it
  pika_add_test_and_deps_compile_test(
    "headers" "${subcategory}" ${name} ${ARGN} OBJECT
  )
endfunction(pika_add_headers_compile_test)

function(pika_add_header_tests category)
  set(options NOLIBS)
  set(one_value_args HEADER_ROOT)
  set(multi_value_args HEADERS EXCLUDE EXCLUDE_FROM_ALL DEPENDENCIES)

  cmake_parse_arguments(
    ${category} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  set(_additional_flags)
  if(${category}_NOLIBS)
    set(_additional_flags ${_additional_flags} NOLIBS)
  endif()

  set(all_headers)
  add_custom_target(tests.headers.${category})

  foreach(header ${${category}_HEADERS})

    # skip all headers in directories containing 'detail'
    string(FIND "${header}" "detail" detail_pos)
    list(FIND ${category}_EXCLUDE "${header}" exclude_pos)

    if(${detail_pos} EQUAL -1 AND ${exclude_pos} EQUAL -1)
      # extract relative path of header
      string(REGEX REPLACE "${${category}_HEADER_ROOT}/" "" relpath "${header}")

      # .hpp --> .cpp
      string(REGEX REPLACE ".hpp" ".cpp" full_test_file "${relpath}")
      # remove extension, '/' --> '_'
      string(REGEX REPLACE ".hpp" "_hpp" test_file "${relpath}")
      string(REGEX REPLACE "/" "_" test_name "${test_file}")

      # generate the test
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}
           "#include <${relpath}>\n" "#ifndef PIKA_MAIN_DEFINED\n"
           "int main() { return 0; }\n" "#endif\n"
      )

      set(exclude_all_pos -1)
      list(FIND ${category}_EXCLUDE_FROM_ALL "${header}" exclude_all_pos)
      if(${exclude_all_pos} EQUAL -1)
        set(all_headers ${all_headers} "#include <${relpath}>\n")
      endif()

      get_filename_component(header_dir "${relpath}" DIRECTORY)

      pika_add_headers_compile_test(
        "${category}" ${test_name}
        SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}"
        SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/${header_dir}"
        FOLDER "Tests/Headers/${header_dir}"
        DEPENDENCIES ${${category}_DEPENDENCIES} pika_private_flags
                     pika_public_flags ${_additional_flags}
      )

    endif()
  endforeach()

  set(test_name "all_headers")
  set(all_headers_test_file "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
  file(WRITE ${all_headers_test_file}
       ${all_headers} "#ifndef PIKA_MAIN_DEFINED\n" "int main() { return 0; }\n"
       "#endif\n"
  )

  pika_add_headers_compile_test(
    "${category}" ${test_name}
    SOURCES "${all_headers_test_file}"
    SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}"
    FOLDER "Tests/Headers"
    DEPENDENCIES ${${category}_DEPENDENCIES} ${_additional_flags}
  )
endfunction()
