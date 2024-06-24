//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PIKA_HAVE_MODULE)

#include <pika/config.hpp>
#include <pika/logging.hpp>
#include <pika/string_util/from_string.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace pika::detail {
    PIKA_DETAIL_DEFINE_SPDLOG(pika, warn)

    spdlog::level::level_enum get_spdlog_level(std::string const& env)
    {
        try
        {
            return static_cast<spdlog::level::level_enum>(
                pika::detail::from_string<std::underlying_type_t<spdlog::level::level_enum>>(env));
        }
        catch (pika::detail::bad_lexical_cast const&)
        {
            fmt::print(std::cerr,
                "pika given invalid log level: \"{}\". Using default level instead {} (warn). "
                "Valid values are {} (trace) to {} (off).\n",
                env, SPDLOG_LEVEL_WARN, SPDLOG_LEVEL_TRACE, SPDLOG_LEVEL_OFF);
            return spdlog::level::warn;
        }
    }

    std::shared_ptr<spdlog::sinks::sink> get_spdlog_sink(std::string const& env)
    {
        if (env.empty())
        {
            fmt::print(
                std::cerr, "pika given empty log destination. Using default instead (cerr).\n");
        }
        else if (env == "cout") { return std::make_shared<spdlog::sinks::stdout_color_sink_mt>(); }
        else if (env == "cerr") { return std::make_shared<spdlog::sinks::stderr_color_sink_mt>(); }
        return std::make_shared<spdlog::sinks::basic_file_sink_mt>(env);
    }
}    // namespace pika::detail

#endif
