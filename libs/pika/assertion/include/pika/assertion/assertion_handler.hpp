//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assertion/source_location.hpp>

#include <string>

namespace pika::detail {
    /// The signature for an assertion handler
    using assertion_handler_type = void (*)(
        source_location const& loc, const char* expr, std::string const& msg);

    /// Set the assertion handler to be used within a program. If the handler has been
    /// set already once, the call to this function will be ignored.
    /// \note This function is not thread safe
    PIKA_EXPORT void set_assertion_handler(assertion_handler_type handler);
}    // namespace pika::detail
