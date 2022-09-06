# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION AND NOT PIKA_FIND_PACKAGE)

  if(PIKA_WITH_CXX_STANDARD LESS 20)

    pika_error(
      "PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION requires at least C++20 (PIKA_WITH_CXX_STANDARD is currently ${PIKA_WITH_CXX_STANDARD})"
    )

  endif()

  find_package(p2300 REQUIRED)
  if(NOT PIKA_FIND_PACKAGE)
    target_link_libraries(pika_base_libraries INTERFACE P2300::p2300)
  endif()

endif()
