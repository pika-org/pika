# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_GPU_SUPPORT)
  pika_find_package(whip 0.1.0 REQUIRED)
  if(NOT PIKA_FIND_PACKAGE)
    target_link_libraries(pika_base_libraries INTERFACE whip::whip)
  endif()
endif()
