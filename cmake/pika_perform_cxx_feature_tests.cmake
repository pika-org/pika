# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2013-2016 Agustin Berge
# Copyright (c)      2017 Taeguk Kwon
# Copyright (c)      2020 Giannis Gonidelis
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ##################################################################################################
# C++ feature tests
# ##################################################################################################
function(pika_perform_cxx_feature_tests)
  pika_check_for_cxx11_std_atomic(REQUIRED "pika needs support for C++11 std::atomic")

  # Separately check for 128 bit atomics
  pika_check_for_cxx11_std_atomic_128bit(DEFINITIONS PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)

  pika_check_for_cxx11_std_quick_exit(DEFINITIONS PIKA_HAVE_CXX11_STD_QUICK_EXIT)

  pika_check_for_cxx11_std_shared_ptr_lwg3018(DEFINITIONS PIKA_HAVE_CXX11_STD_SHARED_PTR_LWG3018)

  pika_check_for_c11_aligned_alloc(DEFINITIONS PIKA_HAVE_C11_ALIGNED_ALLOC)

  pika_check_for_cxx17_std_aligned_alloc(DEFINITIONS PIKA_HAVE_CXX17_STD_ALIGNED_ALLOC)

  pika_check_for_cxx17_aligned_new(DEFINITIONS PIKA_HAVE_CXX17_ALIGNED_NEW)

  pika_check_for_cxx17_copy_elision(DEFINITIONS PIKA_HAVE_CXX17_COPY_ELISION)

  pika_check_for_cxx17_memory_resource(DEFINITIONS PIKA_HAVE_CXX17_MEMORY_RESOURCE)

  # C++20 feature tests
  pika_check_for_cxx20_no_unique_address_attribute(
    DEFINITIONS PIKA_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
  )

  pika_check_for_cxx20_std_disable_sized_sentinel_for(
    DEFINITIONS PIKA_HAVE_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
  )

  pika_check_for_cxx20_trivial_virtual_destructor(
    DEFINITIONS PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR
  )

  pika_check_for_cxx20_trivial_virtual_destructor_gpu(
    DEFINITIONS PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR_GPU
  )

  pika_check_for_cxx23_static_call_operator(DEFINITIONS PIKA_HAVE_CXX23_STATIC_CALL_OPERATOR)

  pika_check_for_cxx23_static_call_operator_gpu(
    DEFINITIONS PIKA_HAVE_CXX23_STATIC_CALL_OPERATOR_GPU
  )

  pika_check_for_cxx_lambda_capture_decltype(DEFINITIONS PIKA_HAVE_CXX_LAMBDA_CAPTURE_DECLTYPE)

  pika_check_for_stdexec_sender_receiver_concepts(
    DEFINITIONS PIKA_HAVE_STDEXEC_SENDER_RECEIVER_CONCEPTS
  )

  pika_check_for_stdexec_continues_on(DEFINITIONS PIKA_HAVE_STDEXEC_CONTINUES_ON)

  pika_check_for_stdexec_env(DEFINITIONS PIKA_HAVE_STDEXEC_ENV)

  pika_check_for_pthread_setname_np(DEFINITIONS PIKA_HAVE_PTHREAD_SETNAME_NP)
endfunction()
