//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/logging/format/named_write.hpp>
#include <pika/logging/level.hpp>
#include <pika/logging/manipulator.hpp>
#include <pika/logging/message.hpp>
#include <pika/modules/logging.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>

#include <string>

#if defined(PIKA_HAVE_LOGGING)
namespace pika::detail {

    /// \cond NOINTERNAL

    // custom log destination: send generated strings to console
    struct PIKA_EXPORT console_local : pika::util::logging::destination::manipulator
    {
        console_local(pika::util::logging::level level, pika::logging_destination dest)
          : level_(level)
          , dest_(dest)
        {
        }

        void operator()(pika::util::logging::message const& msg) override;

        friend bool operator==(console_local const& lhs, console_local const& rhs)
        {
            return lhs.dest_ == rhs.dest_;
        }

        pika::util::logging::level level_;
        pika::logging_destination dest_;
    };

    struct log_settings
    {
        std::string level_;
        std::string dest_;
        std::string format_;
    };

    PIKA_EXPORT void define_common_formatters(pika::util::logging::writer::named_write& writer);

    PIKA_EXPORT void define_formatters_local(pika::util::logging::writer::named_write& writer);

    PIKA_EXPORT log_settings get_log_settings(pika::detail::section const&, char const*);
    PIKA_EXPORT void init_logging(pika::util::runtime_configuration& ini, bool isconsole,
        void (*set_console_dest)(pika::util::logging::writer::named_write&, char const*,
            pika::util::logging::level, pika::logging_destination),
        void (*define_formatters)(pika::util::logging::writer::named_write&));

    PIKA_EXPORT void init_logging_local(pika::util::runtime_configuration&);

    /// \endcond

    /// Enable logging for given destination
    PIKA_EXPORT void enable_logging(pika::logging_destination dest, std::string const& lvl = "5",
        std::string logdest = "", std::string logformat = "");

    /// Disable all logging for the given destination
    PIKA_EXPORT void disable_logging(pika::logging_destination dest);
}    // namespace pika::detail

#else    // PIKA_HAVE_LOGGING

namespace pika::detail {
    PIKA_EXPORT void warn_if_logging_requested(pika::util::runtime_configuration&);

    PIKA_EXPORT void enable_logging(pika::logging_destination dest, std::string const& lvl = "5",
        std::string logdest = "", std::string logformat = "");

    PIKA_EXPORT void disable_logging(pika::logging_destination dest);
}    // namespace pika::detail

#endif    // PIKA_HAVE_LOGGING
