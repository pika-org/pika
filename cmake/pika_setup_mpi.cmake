# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME : in the future put it directly inside the cmake directory of the
# corresponding plugin

include(pika_message)

macro(pika_setup_mpi)
  if(NOT TARGET Mpi::mpi)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(Mpi::mpi INTERFACE IMPORTED)
    target_link_libraries(Mpi::mpi INTERFACE MPI::MPI_CXX)
    # Ensure compatibility with older versions
    if(MPI_LIBRARY)
      target_link_libraries(Mpi::mpi INTERFACE ${MPI_LIBRARY})
    endif()
    if(MPI_EXTRA_LIBRARY)
      target_link_libraries(Mpi::mpi INTERFACE ${MPI_EXTRA_LIBRARY})
    endif()

    pika_info("MPI version: " ${MPI_CXX_VERSION})
  endif()
endmacro()

if(PIKA_WITH_MPI)
  pika_setup_mpi()
endif()
