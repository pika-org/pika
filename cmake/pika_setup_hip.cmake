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
  find_package(hipblas HINTS $ENV{HIPBLAS_ROOT} CONFIG)
  if(NOT hipblas_FOUND)
    pika_warn(
      "Hipblas could not be found, the blas parts will therefore be disabled.\n\
      You can reconfigure specifying HIPBLAS_ROOT to enable hipblas"
    )
    set(PIKA_WITH_GPUBLAS OFF)
  else()
    set(PIKA_WITH_GPUBLAS ON)
    pika_add_config_define(PIKA_HAVE_GPUBLAS)
  endif()

  if(NOT PIKA_FIND_PACKAGE)
    # The cmake variables are supposed to be cached no need to redefine them
    pika_add_config_define(PIKA_HAVE_HIP)
  endif()

endif()
