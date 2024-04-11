// macros.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

// IMPORTANT : the JT28092007_macros_HPP_DEFINED needs to remain constant
// - don't change the macro name!
#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace pika::util::logging {
// TODO: Return ptr directly? Return reference?
#define PIKA_DETAIL_DECLARE_SPDLOG(name)                                                           \
    std::shared_ptr<spdlog::logger> get_##name##_logger() noexcept;

#define PIKA_DETAIL_DEFINE_SPDLOG(name, loglevel)                                                  \
    std::shared_ptr<spdlog::logger> get_##name##_logger() noexcept                                 \
    {                                                                                              \
        static auto logger = []() {                                                                \
            auto logger = std::make_shared<spdlog::logger>(                                        \
                #name, std::make_shared<spdlog::sinks::stdout_color_sink_mt>());                   \
            logger->set_level(spdlog::level::loglevel);                                            \
            return logger;                                                                         \
        }();                                                                                       \
        return logger;                                                                             \
    }

#define PIKA_DETAIL_SPDLOG(name, loglevel, ...)                                                    \
    if (::pika::util::get_##name##_logger()->level() <= spdlog::level::loglevel)                   \
    {                                                                                              \
        ::pika::util::get_##name##_logger()->log(                                                  \
            spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::loglevel,      \
            __VA_ARGS__);                                                                          \
    }

#define PIKA_DETAIL_SPDLOG_ENABLED(name, loglevel)                                                 \
    (::pika::util::get_##name##_logger()->level() <= spdlog::level::loglevel)
}    // namespace pika::util::logging
