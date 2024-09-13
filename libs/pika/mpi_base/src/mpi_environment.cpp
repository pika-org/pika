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

#include <boost/tokenizer.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {

    namespace detail {

        bool detect_mpi_environment(util::runtime_configuration const& cfg, char const* default_env)
        {
#if defined(__bgq__)
            // If running on BG/Q, we can safely assume to always run in an MPI environment
            return true;
#endif
            std::string mpi_environment_strings = cfg.get_entry("pika.mpi.env", default_env);

            boost::char_separator<char> sep(";,: ");
            boost::tokenizer<boost::char_separator<char>> tokens(mpi_environment_strings, sep);
            for (auto const& tok : tokens)
            {
                char* env = std::getenv(tok.c_str());
                if (env)
                {
                    PIKA_LOG(debug, "Found MPI environment variable: {} = {}, enabling MPI support",
                        tok, std::string(env));
                    return true;
                }
            }

            PIKA_LOG(info, "No known MPI environment variable found, no MPI support");
            return false;
        }
    }    // namespace detail

    bool mpi_environment::check_mpi_environment(util::runtime_configuration const& cfg)
    {
        // log message was already generated
        return detail::detect_mpi_environment(cfg, PIKA_HAVE_MPI_ENV);
    }
}}    // namespace pika::util

#if defined(PIKA_HAVE_MODULE_MPI_BASE)

namespace pika { namespace util {

    bool mpi_environment::mpi_init_pika_ = false;

    ///////////////////////////////////////////////////////////////////////////
    int mpi_environment::init(int*, char***, const int required, const int minimal, int& provided)
    {
        mpi_init_pika_ = false;
        int retval = MPI_SUCCESS;
        if (!mpi_environment::is_mpi_inititialized())
        {
            retval = MPI_Init_thread(nullptr, nullptr, required, &provided);
            if (MPI_SUCCESS != retval) { return retval; }

            if (provided < minimal)
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                    "pika::util::mpi_environment::init",
                    "MPI doesn't provide minimal requested thread level");
            }
            mpi_init_pika_ = true;
        }
        return retval;
    }

    bool mpi_environment::is_mpi_inititialized()
    {
        int is_initialized = 0;
        int retval = MPI_Initialized(&is_initialized);
        if (MPI_SUCCESS != retval) { throw std::runtime_error("MPI_Initialized call failed"); }
        return is_initialized;
    }

    std::string mpi_environment::get_processor_name()
    {
        char name[MPI_MAX_PROCESSOR_NAME + 1] = {'\0'};
        int len = 0;
        MPI_Get_processor_name(name, &len);

        return name;
    }

    void mpi_environment::finalize()
    {
        if (pika_called_init())
        {
            int is_finalized = 0;
            MPI_Finalized(&is_finalized);
            if (!is_finalized) { MPI_Finalize(); }
        }
    }

    bool mpi_environment::pika_called_init() { return mpi_init_pika_; }

    int mpi_environment::size()
    {
        int res(-1);
        MPI_Comm_size(MPI_COMM_WORLD, &res);
        return res;
    }

    int mpi_environment::rank()
    {
        int res(-1);
        MPI_Comm_rank(MPI_COMM_WORLD, &res);
        return res;
    }
}}    // namespace pika::util

#endif
