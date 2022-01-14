//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/modules/logging.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>

#include <string>

#if defined(PIKA_HAVE_LOGGING)
///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    struct PIKA_EXPORT console_local : logging::destination::manipulator
    {
        console_local(logging::level level, logging_destination dest)
          : level_(level)
          , dest_(dest)
        {
        }

        void operator()(logging::message const& msg) override;

        friend bool operator==(
            console_local const& lhs, console_local const& rhs)
        {
            return lhs.dest_ == rhs.dest_;
        }

        logging::level level_;
        logging_destination dest_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        struct log_settings
        {
            std::string level_;
            std::string dest_;
            std::string format_;
        };

        PIKA_EXPORT void define_common_formatters(
            logging::writer::named_write& writer);

        PIKA_EXPORT void define_formatters_local(
            logging::writer::named_write& writer);

        PIKA_EXPORT log_settings get_log_settings(
            util::section const&, char const*);

        PIKA_EXPORT void init_logging(runtime_configuration& ini,
            bool isconsole,
            void (*set_console_dest)(logging::writer::named_write&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&));

        PIKA_EXPORT void init_logging_local(runtime_configuration&);
    }    // namespace detail

    /// \endcond

    //////////////////////////////////////////////////////////////////////////
    /// Enable logging for given destination
    PIKA_EXPORT void enable_logging(logging_destination dest,
        std::string const& lvl = "5", std::string logdest = "",
        std::string logformat = "");

    /// Disable all logging for the given destination
    PIKA_EXPORT void disable_logging(logging_destination dest);
}}    // namespace pika::util

#else    // PIKA_HAVE_LOGGING

namespace pika { namespace util {
    namespace detail {

        PIKA_EXPORT void warn_if_logging_requested(runtime_configuration&);
    }

    //////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT void enable_logging(logging_destination dest,
        std::string const& lvl = "5", std::string logdest = "",
        std::string logformat = "");

    PIKA_EXPORT void disable_logging(logging_destination dest);
}}    // namespace pika::util

#endif    // PIKA_HAVE_LOGGING
