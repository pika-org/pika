//  Copyright (c) 2024 ETH Zurich
//  Copyright (c) 2013-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>

#include <pika/logging.hpp>
#include <pika/modules/mpi_base.hpp>
#include <pika/modules/runtime_configuration.hpp>
#include <pika/modules/util.hpp>
#include <pika/mpi_base/mpi_exception.hpp>

#include <boost/tokenizer.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>

namespace pika::mpi::detail {

    bool environment::mpi_init_pika_ = false;

    ///////////////////////////////////////////////////////////////////////////
    int environment::init(int*, char***, const int required, const int minimal, int& provided)
    {
        mpi_init_pika_ = false;
        int retval = MPI_SUCCESS;
        if (!environment::is_mpi_initialized())
        {
            retval = MPI_Init_thread(nullptr, nullptr, required, &provided);
            if (MPI_SUCCESS != retval) { return retval; }

            if (provided < minimal)
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::mpi::environment::init",
                    "MPI doesn't provide minimal requested thread level");
            }
            mpi_init_pika_ = true;
        }
        return retval;
    }

    bool environment::is_mpi_initialized()
    {
        int is_initialized = 0;
        int retval = MPI_Initialized(&is_initialized);
        if (MPI_SUCCESS != retval)
        {
            throw mpi::exception(MPI_ERR_OTHER, "MPI_Initialized call failed");
        }
        return is_initialized;
    }

    std::string environment::get_processor_name()
    {
        char name[MPI_MAX_PROCESSOR_NAME + 1] = {'\0'};
        int len = 0;
        MPI_Get_processor_name(name, &len);

        return name;
    }

    void environment::finalize()
    {
        if (pika_called_init())
        {
            int is_finalized = 0;
            MPI_Finalized(&is_finalized);
            if (!is_finalized) { MPI_Finalize(); }
        }
    }

    bool environment::pika_called_init() { return mpi_init_pika_; }

    int environment::size()
    {
        int res(-1);
        MPI_Comm_size(MPI_COMM_WORLD, &res);
        return res;
    }

    int environment::rank()
    {
        int res(-1);
        MPI_Comm_rank(MPI_COMM_WORLD, &res);
        return res;
    }
}    // namespace pika::mpi::detail
