//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config/export_definitions.hpp>

#include <iosfwd>

namespace pika::detail {
    /// This contains the location information where \a PIKA_ASSERT has been
    /// called
    struct source_location
    {
        char const* file_name;
        unsigned line_number;
        char const* function_name;
    };
    PIKA_EXPORT std::ostream& operator<<(std::ostream& os, source_location const& loc);
}    // namespace pika::detail
