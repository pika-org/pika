# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(pika_message)
include(pika_add_pseudo_dependencies)
include(pika_add_pseudo_target)

# add example pseudo targets needed for modules
if(PIKA_WITH_EXAMPLES)
  pika_add_pseudo_target(examples.modules)
  pika_add_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules
# pika_warn("PIKA_WITH_TESTS=${PIKA_WITH_TESTS}")
if(PIKA_WITH_TESTS)
  if(PIKA_WITH_TESTS_UNIT)
    pika_add_pseudo_target(tests.unit.modules)
    pika_add_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()

  if(PIKA_WITH_EXAMPLES AND PIKA_WITH_TESTS_EXAMPLES)
    pika_add_pseudo_target(tests.examples.modules)
    pika_add_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  if(PIKA_WITH_TESTS_REGRESSIONS)
    pika_add_pseudo_target(tests.regressions.modules)
    pika_add_pseudo_dependencies(
      tests.regressions tests.regressions.modules
    )
  endif()

  if(PIKA_WITH_TESTS_BENCHMARKS)
    pika_add_pseudo_target(tests.performance.modules)
    pika_add_pseudo_dependencies(
      tests.performance tests.performance.modules
    )
  endif()

  if(PIKA_WITH_TESTS_HEADERS)
    pika_add_pseudo_target(tests.headers.modules)
    pika_add_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()

pika_info("")
pika_info("Configuring modules:")

# Generate a file that lists all enabled modules for checks that might be
# optional based on the presence of a module or not
set(MODULE_ENABLED_MODULE_DEFINES)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

if(MSVC AND MSVC14)
  # add natvis files to solution
  pika_add_library_sources(
    pika_natvis_files GLOB GLOBS "${PROJECT_SOURCE_DIR}/tools/VS/*.natvis"
  )
  source_group("Natvis Files" FILES ${pika_natvis_files_SOURCES})
endif()

set(pika_libs pika)

foreach(lib ${pika_libs})
  add_library(${lib} SHARED src/dummy.cpp ${pika_natvis_files_SOURCES})
  set_target_properties(
    ${lib}
    PROPERTIES VERSION ${PIKA_VERSION}
               SOVERSION ${PIKA_SOVERSION}
               CLEAN_DIRECT_OUTPUT 1
               OUTPUT_NAME ${lib}
               FOLDER "Core"
  )
  install(
    TARGETS ${lib}
    EXPORT pika_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${lib}
  )
  pika_export_targets(${lib})

  add_subdirectory(${lib})
endforeach()

# pika_add_module populates PIKA_ENABLED_MODULES
foreach(module ${PIKA_ENABLED_MODULES})
  string(TOUPPER ${module} uppercase_module)

  set(MODULE_ENABLED_MODULE_DEFINES
      "${MODULE_ENABLED_MODULE_DEFINES}#define PIKA_HAVE_MODULE_${uppercase_module}\n"
  )
endforeach()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/modules_enabled.hpp.in"
  "${PROJECT_BINARY_DIR}/libs/pika/config_local/include/pika/local/config/modules_enabled.hpp"
  @ONLY
)

if(MSVC AND PIKA_COROUTINES_WITH_SWAP_CONTEXT_EMULATION)
  target_link_options(pika PRIVATE "/EXPORT:switch_to_fiber")
endif()

if(PIKA_WITH_DATAPAR_VC)
  target_link_libraries(pika PUBLIC Vc::vc)
endif()

if(PIKA_WITH_ITTNOTIFY)
  target_link_libraries(pika PUBLIC Amplifier::amplifier)
endif()
