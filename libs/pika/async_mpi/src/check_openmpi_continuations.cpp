// simple compile test to see if mpix continuuations are present
// will return 0 when available,
// will return 1 when available

#include <mpi.h>

#if __has_include(<mpi-ext.h>)
# include <mpi-ext.h>
#endif

#ifdef OMPI_HAVE_MPI_EXT_CONTINUE
int main() { return 0; }
#else
int main() { return 1; }
#endif
