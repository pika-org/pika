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
    };
}    // namespace pika

#include <pika/assertion/current_function.hpp>
#include <pika/logging/level.hpp>
#include <pika/logging/logging.hpp>

#include <string>

////////////////////////////////////////////////////////////////////////////////
#define PIKA_LOG(loglevel, ...) PIKA_DETAIL_LOG_PIKA(loglevel, __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////
namespace pika::util {

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {
        PIKA_EXPORT pika::util::logging::level get_log_level(
            std::string const& env, bool allow_always = false);
        PIKA_EXPORT spdlog::level::level_enum get_spdlog_level(std::string const& env);
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT PIKA_DETAIL_DECLARE_SPDLOG(pika)
#define LPIKA_ENABLED(loglevel) PIKA_DETAIL_SPDLOG_ENABLED(pika, loglevel)
#define PIKA_DETAIL_LOG_PIKA(loglevel, ...) PIKA_DETAIL_SPDLOG(pika, loglevel, __VA_ARGS__)

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
    PIKA_LOG(info, "{}", t);
    return l;
}

constexpr bootstrap_logging lbt_;
