# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_STDEXEC)

  if(PIKA_WITH_CXX_STANDARD LESS 20)

    pika_error(
      "PIKA_WITH_STDEXEC requires at least C++20 (PIKA_WITH_CXX_STANDARD is currently ${PIKA_WITH_CXX_STANDARD})"
    )

  endif()

  pika_find_package(STDEXEC REQUIRED)
  if(NOT PIKA_FIND_PACKAGE)
    target_link_libraries(pika_base_libraries INTERFACE STDEXEC::stdexec)
  endif()

  if(NOT PIKA_FIND_PACKAGE)
    get_target_property(_stdexec_include_dirs STDEXEC::stdexec INTERFACE_INCLUDE_DIRECTORIES)

    pika_check_for_stdexec_sender_receiver_concepts(
      DEFINITIONS PIKA_HAVE_STDEXEC_SENDER_RECEIVER_CONCEPTS INCLUDE_DIRECTORIES
      ${_stdexec_include_dirs}
    )
    pika_check_for_stdexec_continues_on(
      DEFINITIONS PIKA_HAVE_STDEXEC_CONTINUES_ON INCLUDE_DIRECTORIES ${_stdexec_include_dirs}
    )
    pika_check_for_stdexec_env(
      DEFINITIONS PIKA_HAVE_STDEXEC_ENV INCLUDE_DIRECTORIES ${_stdexec_include_dirs}
    )
    pika_check_for_stdexec_member_queries(
      DEFINITIONS PIKA_HAVE_STDEXEC_MEMBER_QUERIES INCLUDE_DIRECTORIES ${_stdexec_include_dirs}
    )
    pika_check_for_stdexec_transform_completion_signatures_of(
      DEFINITIONS PIKA_HAVE_STDEXEC_TRANSFORM_COMPLETION_SIGNATURES_OF INCLUDE_DIRECTORIES
      ${_stdexec_include_dirs}
    )
  endif()

endif()
