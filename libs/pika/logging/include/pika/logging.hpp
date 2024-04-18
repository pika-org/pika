//  Copyright (c)      2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace pika::detail {
// TODO: Return ptr directly? Return reference?
#define PIKA_DETAIL_DECLARE_SPDLOG(name)                                                           \
    std::shared_ptr<spdlog::logger> get_##name##_logger() noexcept;

#define PIKA_DETAIL_DEFINE_SPDLOG(name, loglevel)                                                  \
    std::shared_ptr<spdlog::logger> get_##name##_logger() noexcept                                 \
    {                                                                                              \
        static auto logger = []() {                                                                \
            auto logger = std::make_shared<spdlog::logger>(                                        \
                #name, std::make_shared<spdlog::sinks::stderr_color_sink_mt>());                   \
            logger->set_level(spdlog::level::loglevel);                                            \
            return logger;                                                                         \
        }();                                                                                       \
        return logger;                                                                             \
    }

#define PIKA_DETAIL_SPDLOG(name, loglevel, ...)                                                    \
    if (::pika::detail::get_##name##_logger()->level() <= spdlog::level::loglevel)                 \
    {                                                                                              \
        ::pika::detail::get_##name##_logger()->log(                                                \
            spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::loglevel,      \
            __VA_ARGS__);                                                                          \
    }

#define PIKA_DETAIL_SPDLOG_ENABLED(name, loglevel)                                                 \
    (::pika::detail::get_##name##_logger()->level() <= spdlog::level::loglevel)
}    // namespace pika::detail

#define PIKA_LOG(loglevel, ...) PIKA_DETAIL_LOG_PIKA(loglevel, __VA_ARGS__)
#define LPIKA_ENABLED(loglevel) PIKA_DETAIL_SPDLOG_ENABLED(pika, loglevel)
#define PIKA_DETAIL_LOG_PIKA(loglevel, ...) PIKA_DETAIL_SPDLOG(pika, loglevel, __VA_ARGS__)

namespace pika::detail {
    PIKA_EXPORT spdlog::level::level_enum get_spdlog_level(std::string const& env);
    PIKA_EXPORT std::shared_ptr<spdlog::sinks::sink> get_spdlog_sink(std::string const& env);
    PIKA_EXPORT PIKA_DETAIL_DECLARE_SPDLOG(pika)
}    // namespace pika::detail