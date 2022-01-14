# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2013-2016 Agustin Berge
# Copyright (c)      2017 Taeguk Kwon
# Copyright (c)      2020 Giannis Gonidelis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ##############################################################################
# C++ feature tests
# ##############################################################################
function(pika_perform_cxx_feature_tests)
  pika_check_for_cxx11_std_atomic(
    REQUIRED "pika needs support for C++11 std::atomic"
  )

  # Separately check for 128 bit atomics
  pika_check_for_cxx11_std_atomic_128bit(
    DEFINITIONS PIKA_HAVE_CXX11_STD_ATOMIC_128BIT
  )

  pika_check_for_cxx11_std_quick_exit(
    DEFINITIONS PIKA_HAVE_CXX11_STD_QUICK_EXIT
  )

  pika_check_for_cxx11_std_shared_ptr_lwg3018(
    DEFINITIONS PIKA_HAVE_CXX11_STD_SHARED_PTR_LWG3018
  )

  pika_check_for_c11_aligned_alloc(DEFINITIONS PIKA_HAVE_C11_ALIGNED_ALLOC)

  pika_check_for_cxx17_std_aligned_alloc(
    DEFINITIONS PIKA_HAVE_CXX17_STD_ALIGNED_ALLOC
  )

  pika_check_for_cxx17_std_execution_policies(
    DEFINITIONS PIKA_HAVE_CXX17_STD_EXECUTION_POLICES
  )

  pika_check_for_cxx17_filesystem(DEFINITIONS PIKA_HAVE_CXX17_FILESYSTEM)

  pika_check_for_cxx17_hardware_destructive_interference_size(
    DEFINITIONS PIKA_HAVE_CXX17_HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE
  )

  pika_check_for_cxx17_aligned_new(DEFINITIONS PIKA_HAVE_CXX17_ALIGNED_NEW)

  pika_check_for_cxx17_shared_ptr_array(
    DEFINITIONS PIKA_HAVE_CXX17_SHARED_PTR_ARRAY
  )

  pika_check_for_cxx17_std_transform_scan(
    DEFINITIONS PIKA_HAVE_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS
  )

  pika_check_for_cxx17_std_scan(
    DEFINITIONS PIKA_HAVE_CXX17_STD_SCAN_ALGORITHMS
  )

  pika_check_for_cxx17_copy_elision(
    DEFINITIONS PIKA_HAVE_CXX17_COPY_ELISION
  )

  pika_check_for_cxx17_memory_resource(
    DEFINITIONS PIKA_HAVE_CXX17_MEMORY_RESOURCE
  )

  # C++20 feature tests
  pika_check_for_cxx20_coroutines(DEFINITIONS PIKA_HAVE_CXX20_COROUTINES)

  pika_check_for_cxx20_experimental_simd(
    DEFINITIONS PIKA_HAVE_CXX20_EXPERIMENTAL_SIMD PIKA_HAVE_DATAPAR
  )

  pika_check_for_cxx20_lambda_capture(
    DEFINITIONS PIKA_HAVE_CXX20_LAMBDA_CAPTURE
  )

  pika_check_for_cxx20_perfect_pack_capture(
    DEFINITIONS PIKA_HAVE_CXX20_PERFECT_PACK_CAPTURE
  )

  pika_check_for_cxx20_no_unique_address_attribute(
    DEFINITIONS PIKA_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
  )

  pika_check_for_cxx20_paren_initialization_of_aggregates(
    DEFINITIONS PIKA_HAVE_CXX20_PAREN_INITIALIZATION_OF_AGGREGATES
  )

  pika_check_for_cxx20_std_disable_sized_sentinel_for(
    DEFINITIONS PIKA_HAVE_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
  )

  pika_check_for_cxx20_std_endian(DEFINITIONS PIKA_HAVE_CXX20_STD_ENDIAN)

  pika_check_for_cxx20_std_execution_policies(
    DEFINITIONS PIKA_HAVE_CXX20_STD_EXECUTION_POLICES
  )

  pika_check_for_cxx20_std_ranges_iter_swap(
    DEFINITIONS PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP
  )

  pika_check_for_cxx20_trivial_virtual_destructor(
    DEFINITIONS PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR
  )

  pika_check_for_cxx_lambda_capture_decltype(
    DEFINITIONS PIKA_HAVE_CXX_LAMBDA_CAPTURE_DECLTYPE
  )

  # Check the availability of certain C++ builtins
  pika_check_for_builtin_integer_pack(
    DEFINITIONS PIKA_HAVE_BUILTIN_INTEGER_PACK
  )

  pika_check_for_builtin_make_integer_seq(
    DEFINITIONS PIKA_HAVE_BUILTIN_MAKE_INTEGER_SEQ
  )

  if(PIKA_WITH_CUDA)
    pika_check_for_builtin_make_integer_seq_cuda(
      DEFINITIONS PIKA_HAVE_BUILTIN_MAKE_INTEGER_SEQ_CUDA
    )
  endif()

  pika_check_for_builtin_type_pack_element(
    DEFINITIONS PIKA_HAVE_BUILTIN_TYPE_PACK_ELEMENT
  )

  if(PIKA_WITH_CUDA)
    pika_check_for_builtin_type_pack_element_cuda(
      DEFINITIONS PIKA_HAVE_BUILTIN_TYPE_PACK_ELEMENT_CUDA
    )
  endif()

endfunction()
