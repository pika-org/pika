# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_COMPILE_ONLY_TESTS)
  set(compile_tests constexpr_version)

  foreach(compile_test ${compile_tests})
    set(sources ${compile_test}.cpp)

    source_group("Source Files" FILES ${sources})

    pika_add_unit_compile_test(
      "modules.version" ${compile_test}
      SOURCES ${sources}
      FOLDER "Tests/Unit/Modules/Version/CompileOnly"
    )
  endforeach()
endif()
