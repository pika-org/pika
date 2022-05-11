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

  find_package(rocblas REQUIRED HINTS $ENV{ROCBLAS_ROOT})
  find_package(rocsolver REQUIRED HINTS $ENV{ROCSOLVER_ROOT})
  find_package(hipblas REQUIRED HINTS $ENV{HIPBLAS_ROOT} CONFIG)

  if(NOT PIKA_FIND_PACKAGE)
    pika_add_config_define(PIKA_HAVE_HIP)
  endif()
endif()
