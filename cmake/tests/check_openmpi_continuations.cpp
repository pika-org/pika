//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// simple compile test to see if mpix continuuations are present
// returns 0 when available, 0 otherwise

#include <mpi-ext.h>
#include <mpi.h>

#if !defined(OMPI_HAVE_MPI_EXT_CONTINUE)
static_assert(false);
#endif

int main() {}
