# Copyright (c) 2019 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# These are a dummy targets that we add compile flags to. All pika targets should
# link to them.
add_library(pika_private_flags INTERFACE)
add_library(pika_public_flags INTERFACE)

# Set C++ standard
target_compile_features(
  pika_private_flags INTERFACE cxx_std_${PIKA_WITH_CXX_STANDARD}
)
target_compile_features(
  pika_public_flags INTERFACE cxx_std_${PIKA_WITH_CXX_STANDARD}
)

# Set other flags that should always be set

# PIKA_DEBUG must be set without a generator expression as it determines ABI
# compatibility. Projects in Release mode using pika in Debug mode must have
# PIKA_DEBUG set, and projects in Debug mode using pika in Release mode must not
# have PIKA_DEBUG set. PIKA_DEBUG must also not be set by projects using pika.
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  target_compile_definitions(pika_private_flags INTERFACE PIKA_DEBUG)
  target_compile_definitions(pika_public_flags INTERFACE PIKA_DEBUG)
endif()

target_compile_definitions(
  pika_private_flags
  INTERFACE $<$<CONFIG:MinSizeRel>:NDEBUG>
  INTERFACE $<$<CONFIG:Release>:NDEBUG>
  INTERFACE $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)

# Remaining flags are set through the macros in
# cmake/pika_add_compile_flag.cmake

include(pika_export_targets)
# Modules can't link to this if not exported
install(
  TARGETS pika_private_flags
  EXPORT pika_internal_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT pika_private_flags
)
install(
  TARGETS pika_public_flags
  EXPORT pika_internal_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT pika_public_flags
)
pika_export_internal_targets(pika_private_flags)
pika_export_internal_targets(pika_public_flags)
