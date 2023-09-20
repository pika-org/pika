//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/logging/level.hpp>

# include <cstddef>
# include <iomanip>
# include <ostream>
# include <stdexcept>
# include <string>
# include <string_view>

///////////////////////////////////////////////////////////////////////////////
namespace pika::util::logging {
    std::string levelname(level value)
    {
        switch (value)
        {
        case pika::util::logging::level::enable_all: return "<all>";
        case pika::util::logging::level::debug: return "<debug>";
        case pika::util::logging::level::info: return "<info>";
        case pika::util::logging::level::warning: return "<warning>";
        case pika::util::logging::level::error: return "<error>";
        case pika::util::logging::level::fatal: return "<fatal>";
        case pika::util::logging::level::always: return "<always>";
        default: break;
        }

        return '<' + std::to_string(static_cast<int>(value)) + '>';
    }
}    // namespace pika::util::logging
