# Copyright (c)      2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_HIP AND NOT TARGET roc::hipblas)

  if(PIKA_WITH_CUDA)
    pika_error(
      "Both PIKA_WITH_CUDA and PIKA_WITH_HIP are ON. Please choose one of \
    them for pika to work properly"
    )
  endif(PIKA_WITH_CUDA)

  # Needed on rostam
  list(APPEND CMAKE_PREFIX_PATH $ENV{HIP_PATH}/lib/cmake/hip)
  list(APPEND CMAKE_PREFIX_PATH $ENV{DEVICE_LIB_PATH}/cmake/AMDDeviceLibs)
  list(APPEND CMAKE_PREFIX_PATH $ENV{DEVICE_LIB_PATH}/cmake/amd_comgr)
  list(APPEND CMAKE_PREFIX_PATH $ENV{DEVICE_LIB_PATH}/cmake/hsa-runtime64)
  # Setup hipblas (creates roc::hipblas)
  find_package(hipblas REQUIRED HINTS $ENV{HIPBLAS_ROOT} CONFIG)

  if(NOT PIKA_FIND_PACKAGE)
    # The cmake variables are supposed to be cached no need to redefine them
    pika_add_config_define(PIKA_HAVE_HIP)
  endif()

endif()
