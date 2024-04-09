//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

namespace pika {
    enum logging_destination
    {
        destination_pika = 0,
        destination_timing = 1,
        destination_debuglog = 2
    };
}    // namespace pika

#include <pika/assertion/current_function.hpp>
#include <pika/logging/level.hpp>
#include <pika/logging/logging.hpp>

#include <string>

////////////////////////////////////////////////////////////////////////////////
// specific logging
#define PIKA_LTM_(loglevel, ...) PIKA_DETAIL_LOG_PIKA(loglevel, __VA_ARGS__)  /* thread manager */
#define PIKA_LRT_(loglevel, ...) PIKA_DETAIL_LOG_PIKA(loglevel, __VA_ARGS__)  /* runtime support */
#define PIKA_LERR_(loglevel, ...) PIKA_DETAIL_LOG_PIKA(loglevel, __VA_ARGS__) /* exceptions */
#define PIKA_LLCO_(loglevel, ...) PIKA_DETAIL_LOG_PIKA(loglevel, __VA_ARGS__) /* lcos */
#define PIKA_LBT_(loglevel, ...) PIKA_DETAIL_LOG_PIKA(loglevel, __VA_ARGS__)  /* bootstrap */
#define PIKA_DETAIL_LOG_PIKA_TM(loglevel, format_string, ...)                                      \
    PIKA_DETAIL_LOG_PIKA(loglevel, " [TM] " format_string, __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////
namespace pika::util {

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {
        PIKA_EXPORT pika::util::logging::level get_log_level(
            std::string const& env, bool allow_always = false);
        PIKA_EXPORT spdlog::level::level_enum get_spdlog_level(std::string const& env);
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT PIKA_DECLARE_LOG(timing)

#define LTIM_(lvl)                                                                                 \
    PIKA_LOG_FORMAT(pika::util::timing, ::pika::util::logging::level::lvl, "{:>10} ",              \
        ::pika::util::logging::level::lvl) /**/
#define LPROGRESS_                                                                                 \
    PIKA_LOG_FORMAT(pika::util::timing, ::pika::util::logging::level::fatal, " {}:{} {} ",         \
        __FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION) /**/

#define LTIM_ENABLED(lvl)                                                                          \
    pika::util::timing_logger()->is_enabled(::pika::util::logging::level::lvl) /**/

        ////////////////////////////////////////////////////////////////////////
        PIKA_EXPORT PIKA_DECLARE_LOG(pika)

#define LPIKA_(lvl, cat)                                                                           \
    PIKA_LOG_FORMAT(pika::util::pika, ::pika::util::logging::level::lvl, "{:>10}{}",               \
        ::pika::util::logging::level::lvl, (cat)) /**/

#define LPIKA_ENABLED(lvl)                                                                         \
    pika::util::pika_logger()->is_enabled(::pika::util::logging::level::lvl) /**/

            PIKA_EXPORT PIKA_DETAIL_DECLARE_SPDLOG(pika)

#define PIKA_DETAIL_LOG_PIKA(loglevel, ...) PIKA_DETAIL_SPDLOG(pika, loglevel, __VA_ARGS__)

        ////////////////////////////////////////////////////////////////////////
        // special debug logging channel
        PIKA_EXPORT PIKA_DECLARE_LOG(debuglog)

#define LDEB_                                                                                      \
    PIKA_LOG_FORMAT(pika::util::debuglog, ::pika::util::logging::level::error, "{:>10} ",          \
        ::pika::util::logging::level::error) /**/

#define LDEB_ENABLED                                                                               \
    pika::util::debuglog_logger()->is_enabled(::pika::util::logging::level::error) /**/

        ////////////////////////////////////////////////////////////////////////
        // errors are logged in a special manner (always to cerr and additionally,
        // if enabled to 'normal' logging destination as well)
        PIKA_EXPORT PIKA_DECLARE_LOG(pika_error)

#define LFATAL_                                                                                    \
    PIKA_LOG_FORMAT(pika::util::pika_error, ::pika::util::logging::level::fatal, "{:>10} [ERR] ",  \
        ::pika::util::logging::level::fatal) /**/
}    // namespace pika::util

// helper type to forward logging during bootstrap to two destinations
struct bootstrap_logging
{
    constexpr bootstrap_logging() {}
};

template <typename T>
bootstrap_logging const& operator<<(bootstrap_logging const& l, T const& t)
{
    // NOLINTNEXTLINE(bugprone-branch-clone)
    PIKA_LBT_(info, "{}", t);
    LPROGRESS_ << t;
    return l;
}

constexpr bootstrap_logging lbt_;
