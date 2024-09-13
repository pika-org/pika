//  Copyright (c) 2013-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_MODULE_MPI_BASE)

# include <pika/modules/runtime_configuration.hpp>
# include <pika/mpi_base/mpi.hpp>

# include <cstdlib>
# include <string>

# include <pika/config/warnings_prefix.hpp>

namespace pika { namespace util {
        struct PIKA_EXPORT mpi_environment
        {
            // This function checks a list of environment variables to see if commonly set
            // ones such as mpich, openmpi, slurm, etc are present.
            // The list of env vars can be set via pika command line params,
            // or defaults used from cmake configure time.
            // Although this function is not currently used in pika,
            // it is left for potential future use
            static bool check_mpi_environment(runtime_configuration const& cfg);

            // calls mpi_init_thread with the thread level requested and reports
            // any problem if the same level is not granted
            static int init(
                int* argc, char*** argv, const int required, const int minimal, int& provided);

            // finalize mpi, do not call unless init was previously called
            static void finalize();

            // returns true if mpi_environment::init has previously been called
            static bool pika_called_init();

            // convenience functions that retrieve mpi settings
            static bool is_mpi_inititialized();
            static int rank();
            static int size();
            static std::string get_processor_name();

        private:
            static bool mpi_init_pika_;
        };
}}    // namespace pika::util

# include <pika/config/warnings_suffix.hpp>

#else

# include <pika/modules/runtime_configuration.hpp>

# include <pika/config/warnings_prefix.hpp>

namespace pika { namespace util {
        struct PIKA_EXPORT mpi_environment
        {
            static bool check_mpi_environment(runtime_configuration const& cfg);
        };
}}    // namespace pika::util

# include <pika/config/warnings_suffix.hpp>

#endif
