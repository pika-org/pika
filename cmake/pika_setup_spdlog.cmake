# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# TODO: minimum version
pika_find_package(spdlog REQUIRED)

if(NOT PIKA_FIND_PACKAGE)
  target_link_libraries(pika_base_libraries INTERFACE spdlog::spdlog)
endif()
