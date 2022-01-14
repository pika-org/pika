//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/runtime_local/report_error.hpp

#pragma once

#include <pika/local/config.hpp>

#include <cstddef>
#include <exception>

namespace pika {
    /// The function report_error reports the given exception to the console
    PIKA_EXPORT void report_error(
        std::size_t num_thread, std::exception_ptr const& e);

    /// The function report_error reports the given exception to the console
    PIKA_EXPORT void report_error(std::exception_ptr const& e);
}    // namespace pika
