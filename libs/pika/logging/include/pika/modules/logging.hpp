//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

namespace pika {
    enum logging_destination
    {
        destination_pika = 0,
        destination_timing = 1,
        destination_app = 2,
        destination_debuglog = 3
    };
}    // namespace pika

#if defined(PIKA_HAVE_LOGGING)

#include <pika/assertion/current_function.hpp>
#include <pika/logging/level.hpp>
#include <pika/logging/logging.hpp>
#include <pika/modules/format.hpp>

#include <string>

////////////////////////////////////////////////////////////////////////////////
// specific logging
#define LTM_(lvl) LPIKA_(lvl, "  [TM] ")  /* thread manager */
#define LRT_(lvl) LPIKA_(lvl, "  [RT] ")  /* runtime support */
#define LERR_(lvl) LPIKA_(lvl, " [ERR] ") /* exceptions */
#define LLCO_(lvl) LPIKA_(lvl, " [LCO] ") /* lcos */
#define LBT_(lvl) LPIKA_(lvl, "  [BT] ")  /* bootstrap */

////////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {
        PIKA_EXPORT pika::util::logging::level get_log_level(
            std::string const& env, bool allow_always = false);
    }

    ////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT PIKA_DECLARE_LOG(timing)

#define LTIM_(lvl)                                                             \
    PIKA_LOG_FORMAT(pika::util::timing, ::pika::util::logging::level::lvl, "{} ", \
        ::pika::util::logging::level::lvl) /**/
#define LPROGRESS_                                                             \
    PIKA_LOG_FORMAT(pika::util::timing, ::pika::util::logging::level::fatal,      \
        " {}:{} {} ", __FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION) /**/

#define LTIM_ENABLED(lvl)                                                      \
    pika::util::timing_logger()->is_enabled(                                    \
        ::pika::util::logging::level::lvl) /**/

    ////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT PIKA_DECLARE_LOG(pika)

#define LPIKA_(lvl, cat)                                                        \
    PIKA_LOG_FORMAT(pika::util::pika, ::pika::util::logging::level::lvl, "{}{}",   \
        ::pika::util::logging::level::lvl, (cat)) /**/

#define LPIKA_ENABLED(lvl)                                                      \
    pika::util::pika_logger()->is_enabled(::pika::util::logging::level::lvl) /**/

    ////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT PIKA_DECLARE_LOG(app)

#define LAPP_(lvl)                                                             \
    PIKA_LOG_FORMAT(pika::util::app, ::pika::util::logging::level::lvl, "{} ",    \
        ::pika::util::logging::level::lvl) /**/

#define LAPP_ENABLED(lvl)                                                      \
    pika::util::app_logger()->is_enabled(::pika::util::logging::level::lvl) /**/

    ////////////////////////////////////////////////////////////////////////
    // special debug logging channel
    PIKA_EXPORT PIKA_DECLARE_LOG(debuglog)

#define LDEB_                                                                  \
    PIKA_LOG_FORMAT(pika::util::debuglog, ::pika::util::logging::level::error,    \
        "{} ", ::pika::util::logging::level::error) /**/

#define LDEB_ENABLED                                                           \
    pika::util::debuglog_logger()->is_enabled(                                  \
        ::pika::util::logging::level::error) /**/

    ////////////////////////////////////////////////////////////////////////
    // errors are logged in a special manner (always to cerr and additionally,
    // if enabled to 'normal' logging destination as well)
    PIKA_EXPORT PIKA_DECLARE_LOG(pika_error)

#define LFATAL_                                                                \
    PIKA_LOG_FORMAT(pika::util::pika_error, ::pika::util::logging::level::fatal,   \
        "{} [ERR] ", ::pika::util::logging::level::fatal) /**/

    //
    PIKA_EXPORT PIKA_DECLARE_LOG(timing_console)
    //
    PIKA_EXPORT PIKA_DECLARE_LOG(pika_console)
    //
    PIKA_EXPORT PIKA_DECLARE_LOG(app_console)
    // special debug logging channel
    PIKA_EXPORT PIKA_DECLARE_LOG(debuglog_console)
}}    // namespace pika::util

///////////////////////////////////////////////////////////////////////////////
#define LTIM_CONSOLE_(lvl)                                                     \
    PIKA_LOG_USE_LOG(pika::util::timing_console,                                 \
        static_cast<::pika::util::logging::level>(lvl))                         \
    /**/

#define LPIKA_CONSOLE_(lvl)                                                     \
    PIKA_LOG_USE_LOG(                                                           \
        pika::util::pika_console, static_cast<::pika::util::logging::level>(lvl)) \
    /**/

#define LAPP_CONSOLE_(lvl)                                                     \
    PIKA_LOG_USE_LOG(                                                           \
        pika::util::app_console, static_cast<::pika::util::logging::level>(lvl)) \
    /**/

#define LDEB_CONSOLE_                                                          \
    PIKA_LOG_USE_LOG(                                                           \
        pika::util::debuglog_console, ::pika::util::logging::level::error)       \
/**/

// helper type to forward logging during bootstrap to two destinations
struct bootstrap_logging
{
    constexpr bootstrap_logging() {}
};

template <typename T>
bootstrap_logging const& operator<<(bootstrap_logging const& l, T const& t)
{
    // NOLINTNEXTLINE(bugprone-branch-clone)
    LBT_(info) << t;
    LPROGRESS_ << t;
    return l;
}

constexpr bootstrap_logging lbt_;

#else
// logging is disabled all together

namespace pika { namespace util {
    namespace detail {
        struct dummy_log_impl
        {
            constexpr dummy_log_impl() noexcept {}

            template <typename T>
            dummy_log_impl const& operator<<(T&&) const noexcept
            {
                return *this;
            }

            template <typename... Args>
            dummy_log_impl const& format(
                char const*, Args const&...) const noexcept
            {
                return *this;
            }
        };
        constexpr dummy_log_impl dummy_log;
    }    // namespace detail

    // clang-format off

    #define LTIM_(lvl)            if(true) {} else pika::util::detail::dummy_log
    #define LPROGRESS_            if(true) {} else pika::util::detail::dummy_log
    #define LPIKA_(lvl, cat)       if(true) {} else pika::util::detail::dummy_log
    #define LAPP_(lvl)            if(true) {} else pika::util::detail::dummy_log
    #define LDEB_                 if(true) {} else pika::util::detail::dummy_log

    #define LTM_(lvl)             if(true) {} else pika::util::detail::dummy_log
    #define LRT_(lvl)             if(true) {} else pika::util::detail::dummy_log
    #define LOSH_(lvl)            if(true) {} else pika::util::detail::dummy_log
    #define LERR_(lvl)            if(true) {} else pika::util::detail::dummy_log
    #define LLCO_(lvl)            if(true) {} else pika::util::detail::dummy_log
    #define LPCS_(lvl)            if(true) {} else pika::util::detail::dummy_log
    #define LAS_(lvl)             if(true) {} else pika::util::detail::dummy_log
    #define LBT_(lvl)             if(true) {} else pika::util::detail::dummy_log

    #define LFATAL_               if(true) {} else pika::util::detail::dummy_log

    #define LTIM_CONSOLE_(lvl)    if(true) {} else pika::util::detail::dummy_log
    #define LPIKA_CONSOLE_(lvl)    if(true) {} else pika::util::detail::dummy_log
    #define LAPP_CONSOLE_(lvl)    if(true) {} else pika::util::detail::dummy_log
    #define LDEB_CONSOLE_         if(true) {} else pika::util::detail::dummy_log

    #define LTIM_ENABLED(lvl)     (false)
    #define LPIKA_ENABLED(lvl)     (false)
    #define LAPP_ENABLED(lvl)     (false)
    #define LDEB_ENABLED          (false)

    // clang-format on

}}    // namespace pika::util

struct bootstrap_logging
{
    constexpr bootstrap_logging() {}
};
constexpr bootstrap_logging lbt_;

template <typename T>
bootstrap_logging const& operator<<(bootstrap_logging const& l, T&&)
{
    return l;
}

#endif
