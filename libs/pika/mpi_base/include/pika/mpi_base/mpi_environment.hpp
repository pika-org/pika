//  Copyright (c) 2013-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_MODULE_MPI_BASE)

#include <pika/modules/runtime_configuration.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/synchronization/spinlock.hpp>

#include <cstdlib>
#include <string>

#include <pika/local/config/warnings_prefix.hpp>

namespace pika { namespace util {
    struct PIKA_EXPORT mpi_environment
    {
        static bool check_mpi_environment(runtime_configuration const& cfg);

        static int init(int* argc, char*** argv, const int required,
            const int minimal, int& provided);
        static void init(int* argc, char*** argv, runtime_configuration& cfg);
        static void finalize();

        static bool enabled();
        static bool multi_threaded();
        static bool has_called_init();

        static int rank();
        static int size();

        static MPI_Comm& communicator();

        static std::string get_processor_name();

        struct PIKA_EXPORT scoped_lock
        {
            scoped_lock();
            scoped_lock(scoped_lock const&) = delete;
            scoped_lock& operator=(scoped_lock const&) = delete;
            ~scoped_lock();
            void unlock();
        };

        struct PIKA_EXPORT scoped_try_lock
        {
            scoped_try_lock();
            scoped_try_lock(scoped_try_lock const&) = delete;
            scoped_try_lock& operator=(scoped_try_lock const&) = delete;
            ~scoped_try_lock();
            void unlock();
            bool locked;
        };

        typedef pika::lcos::local::spinlock mutex_type;

    private:
        static mutex_type mtx_;

        static bool enabled_;
        static bool has_called_init_;
        static int provided_threading_flag_;
        static MPI_Comm communicator_;

        static int is_initialized_;
    };
}}    // namespace pika::util

#include <pika/local/config/warnings_suffix.hpp>

#else

#include <pika/modules/runtime_configuration.hpp>

#include <pika/local/config/warnings_prefix.hpp>

namespace pika { namespace util {
    struct PIKA_EXPORT mpi_environment
    {
        static bool check_mpi_environment(runtime_configuration const& cfg);
    };
}}    // namespace pika::util

#include <pika/local/config/warnings_suffix.hpp>

#endif
