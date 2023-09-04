# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(GNUInstallDirs)

add_library(pika_omp_hack STATIC src/pika_omp_hack.cpp)
target_link_libraries(pika_omp_hack PRIVATE pika::pika)
install(
  TARGETS pika_omp_hack
  EXPORT pika_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT pika_omp_hack
)
pika_export_targets(pika_omp_hack)
