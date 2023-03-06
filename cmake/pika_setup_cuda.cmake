# Copyright (c) 2019 Ste||ar-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_CUDA AND NOT TARGET Cuda::cuda)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    if(NOT PIKA_FIND_PACKAGE)
      # The cmake variables are supposed to be cached no need to redefine them
      pika_add_config_define(PIKA_HAVE_CUDA)

      if(PIKA_WITH_COMPILER_WARNINGS_AS_ERRORS)
        # target_compile_options( pika_private_flags INTERFACE
        # $<$<AND:$<COMPILE_LANGUAGE:CXX>:$<CXX_COMPILER_ID:NVHPC>>:-Werror>
        # $<$<AND:$<COMPILE_LANGUAGE:CUDA>:$<CUDA_COMPILER_ID:NVHPC>>:-Werror> )
      endif()
    endif()

    find_package(CUDAToolkit MODULE REQUIRED)
    if(NOT PIKA_FIND_PACKAGE)
      target_link_libraries(pika_base_libraries INTERFACE CUDA::cudart)
      target_link_libraries(
        pika_base_libraries INTERFACE CUDA::cublas CUDA::cusolver
      )
    endif()
    add_library(Cuda::cuda INTERFACE IMPORTED)
  else()
    # Check and set CUDA standard
    if(NOT PIKA_FIND_PACKAGE)
      if(DEFINED CMAKE_CUDA_STANDARD AND NOT CMAKE_CUDA_STANDARD STREQUAL
                                         PIKA_WITH_CXX_STANDARD
      )
        pika_error(
          "You've set CMAKE_CUDA_STANDARD to ${CMAKE_CUDA_STANDARD} and PIKA_WITH_CXX_STANDARD to ${PIKA_WITH_CXX_STANDARD}. Please unset CMAKE_CUDA_STANDARD."
        )
      endif()

      set(CMAKE_CUDA_STANDARD ${PIKA_WITH_CXX_STANDARD})
    endif()

    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    enable_language(CUDA)

    if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
      set(PIKA_WITH_CLANG_CUDA ON)
    endif()

    if(NOT PIKA_FIND_PACKAGE)
      # The cmake variables are supposed to be cached no need to redefine them
      pika_add_config_define(PIKA_HAVE_CUDA)
    endif()

    # CUDA libraries used
    add_library(Cuda::cuda INTERFACE IMPORTED)
    # Toolkit targets like CUDA::cudart, CUDA::cublas, CUDA::cufft, etc.
    # available
    find_package(CUDAToolkit MODULE REQUIRED)
    if(CUDAToolkit_FOUND)
      target_link_libraries(Cuda::cuda INTERFACE CUDA::cudart)
      target_link_libraries(Cuda::cuda INTERFACE CUDA::cublas CUDA::cusolver)
    endif()
    # Flag not working for CLANG CUDA
    target_compile_features(
      Cuda::cuda INTERFACE cuda_std_${PIKA_WITH_CXX_STANDARD}
    )
    set_target_properties(
      Cuda::cuda PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON
    )

    target_compile_definitions(
      Cuda::cuda
      INTERFACE
        $<$<AND:$<CUDA_COMPILER_ID:Clang>,$<COMPILE_LANGUAGE:CUDA>>:FMT_USE_FLOAT128=0>
        $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<COMPILE_LANGUAGE:CUDA>>:FMT_USE_NONTYPE_TEMPLATE_ARGS=0>
    )

    if(PIKA_WITH_CLANG_CUDA)
      if(NOT PIKA_FIND_PACKAGE)
        pika_add_target_compile_option(-DBOOST_THREAD_USES_MOVE PUBLIC)
      endif()
    else()
      if(MSVC)
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
          INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>: -DNDEBUG
                    -O2 -Xcompiler=-MD,-Ox -Xcompiler=-bigobj >>
        )
      endif()
      set(CUDA_SEPARABLE_COMPILATION ON)
      target_compile_options(
        Cuda::cuda
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda
                  --default-stream per-thread --expt-relaxed-constexpr >
      )
      if(NOT PIKA_FIND_PACKAGE)
        if(PIKA_WITH_COMPILER_WARNINGS_AS_ERRORS)
          target_compile_options(
            pika_private_flags
            INTERFACE
              $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--Werror
              all-warnings>
              $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:NVHPC>>:-Werror>
              $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVHPC>>:-Werror>
          )
        endif()

        target_compile_options(
          pika_private_flags
          INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--display-error-number>
        )
      endif()
    endif()

    if(NOT PIKA_FIND_PACKAGE)
      target_link_libraries(pika_base_libraries INTERFACE Cuda::cuda)
    endif()
  endif()
endif()
