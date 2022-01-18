# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(NOT PIKA_FIND_PACKAGE)
  target_link_libraries(pika_base_libraries INTERFACE Threads::Threads)

  pika_add_compile_flag_if_available(-pthread PUBLIC)
  pika_add_link_flag_if_available(-pthread PUBLIC)
endif()
