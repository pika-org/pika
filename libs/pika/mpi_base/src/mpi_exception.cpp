//  Copyright (c) 2024 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/errors/exception.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/mpi_base/mpi_exception.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace pika::mpi {
    namespace detail {
        std::string error_message(int code)
        {
            int N = 1023;
            int const len = 1024;
            char buff[len] = {0};
            MPI_Error_string(code, buff, &N);
            return std::string(buff);
        }
    }    // namespace detail

    // -------------------------------------------------------------------------
    // exception type for failed launch of MPI functions or other mpi problem
    exception::exception(int err_code, std::string const& msg)
      : pika::exception(pika::error::bad_function_call,
            msg + std::string(" MPI returned with error: ") + detail::error_message(err_code))
      , err_code_(err_code)
    {
    }

    int exception::get_mpi_errorcode() const noexcept { return err_code_; }
}    // namespace pika::mpi
