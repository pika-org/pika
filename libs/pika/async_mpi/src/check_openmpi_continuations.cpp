//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// simple compile test to see if mpix continuuations are present
// returns 0 when available, 0 otherwise

#include <mpi.h>

#if __has_include(<mpi-ext.h>)
# include <mpi-ext.h>
#endif

#ifdef OMPI_HAVE_MPI_EXT_CONTINUE
int main() { return 0; }
#else
int main() { return 1; }
#endif
