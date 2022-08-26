# Copyright (c)      2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_HIP AND NOT TARGET roc::rocblas)
  if(PIKA_WITH_CUDA)
    pika_error(
      "Both PIKA_WITH_CUDA and PIKA_WITH_HIP are ON. Please choose one of \
    them for pika to work properly"
    )
  endif(PIKA_WITH_CUDA)

  # Check and set HIP standard
  if(NOT PIKA_FIND_PACKAGE)
    if(DEFINED CMAKE_HIP_STANDARD AND NOT CMAKE_HIP_STANDARD STREQUAL
                                      PIKA_WITH_CXX_STANDARD
    )
      pika_error(
        "You've set CMAKE_HIP_STANDARD to ${CMAKE_HIP_STANDARD} and PIKA_WITH_CXX_STANDARD to ${PIKA_WITH_CXX_STANDARD}. Please unset CMAKE_HIP_STANDARD."
      )
    endif()
    set(CMAKE_HIP_STANDARD ${PIKA_WITH_CXX_STANDARD})
  endif()

  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)

  enable_language(HIP)

  find_package(rocblas REQUIRED)
  find_package(rocsolver REQUIRED)
  find_package(hipblas REQUIRED CONFIG)

  if(NOT PIKA_FIND_PACKAGE)
    pika_add_config_define(PIKA_HAVE_HIP)
  endif()
endif()
