//  Copyright (c) 2024 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/errors/exception.hpp>

#include <string>

namespace pika::mpi {
    namespace detail {
        PIKA_EXPORT std::string error_message(int code);
    }    // namespace detail

    struct exception : pika::exception
    {
        PIKA_EXPORT explicit exception(int err_code, const std::string& msg = "");
        PIKA_EXPORT int get_mpi_errorcode() const noexcept;

    protected:
        int err_code_;
    };

}    // namespace pika::mpi
