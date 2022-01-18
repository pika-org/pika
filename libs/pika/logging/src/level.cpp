//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/logging/level.hpp>

#if defined(PIKA_HAVE_LOGGING)

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util { namespace logging {

    static std::string levelname(level value)
    {
        switch (value)
        {
        case pika::util::logging::level::enable_all:
            return "<all>";
        case pika::util::logging::level::debug:
            return "<debug>";
        case pika::util::logging::level::info:
            return "<info>";
        case pika::util::logging::level::warning:
            return "<warning>";
        case pika::util::logging::level::error:
            return "<error>";
        case pika::util::logging::level::fatal:
            return "<fatal>";
        case pika::util::logging::level::always:
            return "<always>";
        default:
            break;
        }

        return '<' + std::to_string(static_cast<int>(value)) + '>';
    }

    void format_value(std::ostream& os, std::string_view spec, level value)
    {
        if (!spec.empty())
            throw std::runtime_error("Not a valid format specifier");

        os << std::right << std::setfill(' ') << std::setw(10)
           << levelname(value);
    }

}}}    // namespace pika::util::logging

#endif    // PIKA_HAVE_LOGGING
