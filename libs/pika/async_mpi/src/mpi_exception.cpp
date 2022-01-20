//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_mpi/mpi_exception.hpp>
#include <pika/errors/exception.hpp>
#include <pika/mpi_base/mpi.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace pika::mpi::experimental {
    namespace detail {
        std::string error_message(int code)
        {
            int N = 1023;
            std::unique_ptr<char[]> err_buff(new char[std::size_t(N) + 1]);
            err_buff[0] = '\0';

            MPI_Error_string(code, err_buff.get(), &N);

            return err_buff.get();
        }

    }    // namespace detail

    // -------------------------------------------------------------------------
    // exception type for failed launch of MPI functions
    mpi_exception::mpi_exception(int err_code, const std::string& msg)
      : pika::exception(pika::bad_function_call,
            msg + std::string(" MPI returned with error: ") +
                detail::error_message(err_code))
      , err_code_(err_code)
    {
    }

    int mpi_exception::get_mpi_errorcode() const noexcept
    {
        return err_code_;
    }
}    // namespace pika::mpi::experimental
