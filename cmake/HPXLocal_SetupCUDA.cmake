# Copyright (c) 2019 Ste||ar-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXLocal_WITH_CUDA AND NOT TARGET Cuda::cuda)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(HPXLocal_WITH_CLANG_CUDA ON)
  endif()

  # Check CUDA standard
  if(DEFINED CMAKE_CUDA_STANDARD AND NOT CMAKE_CUDA_STANDARD STREQUAL
                                     HPXLocal_WITH_CXX_STANDARD
  )
    hpx_local_error(
      "You've set CMAKE_CUDA_STANDARD to ${CMAKE_CUDA_STANDARD} and HPXLocal_WITH_CXX_STANDARD to ${HPXLocal_WITH_CXX_STANDARD}. Please unset CMAKE_CUDA_STANDARD."
    )
  endif()

  set(CMAKE_CUDA_STANDARD ${HPXLocal_WITH_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  enable_language(CUDA)

  if(NOT HPXLocal_FIND_PACKAGE)
    # The cmake variables are supposed to be cached no need to redefine them
    hpx_local_add_config_define(HPX_HAVE_CUDA)
  endif()

  # CUDA libraries used
  add_library(Cuda::cuda INTERFACE IMPORTED)
  # Toolkit targets like CUDA::cudart, CUDA::cublas, CUDA::cufft, etc. available
  find_package(CUDAToolkit MODULE REQUIRED)
  if(CUDAToolkit_FOUND)
    target_link_libraries(Cuda::cuda INTERFACE CUDA::cudart)
    if(TARGET CUDA::cublas)
      set(HPXLocal_WITH_GPUBLAS ON)
      hpx_local_add_config_define(HPX_HAVE_GPUBLAS)
      target_link_libraries(Cuda::cuda INTERFACE CUDA::cublas)
    else()
      set(HPXLocal_WITH_GPUBLAS OFF)
    endif()
  endif()
  # Flag not working for CLANG CUDA
  target_compile_features(Cuda::cuda INTERFACE cuda_std_${CMAKE_CUDA_STANDARD})
  set_target_properties(
    Cuda::cuda PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  if(NOT HPXLocal_WITH_CLANG_CUDA)
    if(NOT MSVC)
      target_compile_options(
        Cuda::cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-w>
      )
    else()
      # Windows
      set(CUDA_PROPAGATE_HOST_FLAGS OFF)
      target_compile_options(
        Cuda::cuda
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:
                  -D_DEBUG
                  -O0
                  -g
                  -G
                  -Xcompiler=-MDd
                  -Xcompiler=-Od
                  -Xcompiler=-Zi
                  -Xcompiler=-bigobj
                  >>
      )
      target_compile_options(
        Cuda::cuda
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:RelWithDebInfo>:
                  -DNDEBUG
                  -O2
                  -g
                  -Xcompiler=-MD,-O2,-Zi
                  -Xcompiler=-bigobj
                  >>
      )
      target_compile_options(
        Cuda::cuda
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:MinSizeRel>: -DNDEBUG
                  -O1 -Xcompiler=-MD,-O1 -Xcompiler=-bigobj >>
      )
      target_compile_options(
        Cuda::cuda
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>: -DNDEBUG -O2
                  -Xcompiler=-MD,-Ox -Xcompiler=-bigobj >>
      )
    endif()
    set(CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(
      Cuda::cuda
      INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --default-stream
                per-thread --expt-relaxed-constexpr >
    )
  else()
    if(NOT HPXLocal_FIND_PACKAGE)
      hpx_local_add_target_compile_option(-DBOOST_THREAD_USES_MOVE PUBLIC)
    endif()
  endif()

  if(NOT HPXLocal_FIND_PACKAGE)
    target_link_libraries(hpx_local_base_libraries INTERFACE Cuda::cuda)
  endif()
endif()
