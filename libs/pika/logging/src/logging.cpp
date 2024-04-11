//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/logging.hpp>
#include <pika/string_util/from_string.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace pika::detail {
    PIKA_DETAIL_DEFINE_SPDLOG(pika, off)

    spdlog::level::level_enum get_spdlog_level(std::string const& env)
    {
        try
        {
            int env_val = pika::detail::from_string<int>(env);
            if (env_val < 0) { return spdlog::level::off; }

            switch (env_val)
            {
                // TODO: Don't invert the log levels
            case 0: return spdlog::level::off;
            case 1: return spdlog::level::critical;
            case 2: return spdlog::level::err;
            case 3: return spdlog::level::warn;
            case 4: return spdlog::level::info;
            case 5: return spdlog::level::debug;
            default: break;
            }
            return spdlog::level::trace;
        }
        catch (pika::detail::bad_lexical_cast const&)
        {
            return spdlog::level::off;
        }
    }
}    // namespace pika::detail
