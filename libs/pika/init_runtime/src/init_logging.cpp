//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>

# include <pika/assert.hpp>
# include <pika/init_runtime/detail/init_logging.hpp>
# include <pika/runtime/get_worker_thread_num.hpp>
# include <pika/runtime_configuration/runtime_configuration.hpp>
# include <pika/threading_base/thread_data.hpp>

# include <fmt/ostream.h>
# include <fmt/printf.h>

# include <cstddef>
# include <cstdint>
# include <cstdlib>
# include <iostream>
# include <string>

namespace pika::detail {
    using logger_writer_type = pika::util::logging::writer::named_write;

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: shepherd
    struct shepherd_thread_id : pika::util::logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::size_t thread_num = pika::get_worker_thread_num();

            if (std::size_t(-1) != thread_num) { fmt::print(to, "{:016x}", thread_num); }
            else { to << std::string(16, '-'); }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: pika thread id
    struct thread_id : pika::util::logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::detail::thread_self* self = threads::detail::get_self_ptr();
            if (nullptr != self)
            {
                // called from inside a pika thread
                threads::detail::thread_id_type id = threads::detail::get_self_id();
                if (id != threads::detail::invalid_thread_id)
                {
                    std::ptrdiff_t value = reinterpret_cast<std::ptrdiff_t>(id.get());
                    fmt::print(to, "{:016x}", value);
                    return;
                }
            }

            // called from outside a pika thread or invalid thread id
            to << std::string(16, '-');
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: pika thread phase
    struct thread_phase : pika::util::logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::detail::thread_self* self = threads::detail::get_self_ptr();
            if (nullptr != self)
            {
                // called from inside a pika thread
                std::size_t phase = self->get_thread_phase();
                if (0 != phase)
                {
                    fmt::print(to, "{:04x}", self->get_thread_phase());
                    return;
                }
            }

            // called from outside a pika thread or no phase given
            to << std::string(4, '-');
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: pika parent thread id
    struct parent_thread_id : pika::util::logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::detail::thread_id_type parent_id = threads::detail::get_parent_id();
            if (nullptr != parent_id)
            {
                // called from inside a pika thread
                std::ptrdiff_t value = reinterpret_cast<std::ptrdiff_t>(parent_id.get());
                fmt::print(to, "{:016x}", value);
            }
            else
            {
                // called from outside a pika thread
                to << std::string(16, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: pika parent thread phase
    struct parent_thread_phase : pika::util::logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::size_t parent_phase = threads::detail::get_parent_phase();
            if (0 != parent_phase)
            {
                // called from inside a pika thread
                fmt::print(to, "{:04x}", parent_phase);
            }
            else
            {
                // called from outside a pika thread
                to << std::string(4, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct dummy_thread_component_id : pika::util::logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override { to << std::string(16, '-'); }
    };

    // unescape config entry
    std::string unescape(std::string const& value)
    {
        std::string result;
        std::string::size_type pos = 0;
        std::string::size_type pos1 = value.find_first_of('\\', 0);
        if (std::string::npos != pos1)
        {
            do {
                switch (value[pos1 + 1])
                {
                case '\\':
                case '\"':
                case '?':
                    result = result + value.substr(pos, pos1 - pos);
                    pos1 = value.find_first_of('\\', (pos = pos1 + 1) + 1);
                    break;

                case 'n':
                    result = result + value.substr(pos, pos1 - pos) + "\n";
                    pos1 = value.find_first_of('\\', pos = pos1 + 1);
                    ++pos;
                    break;

                default:
                    result = result + value.substr(pos, pos1 - pos + 1);
                    pos1 = value.find_first_of('\\', pos = pos1 + 1);
                }

            } while (pos1 != std::string::npos);
            result = result + value.substr(pos);
        }
        else
        {
            // the string doesn't contain any escaped character sequences
            result = value;
        }
        return result;
    }

    void define_common_formatters(logger_writer_type& writer)
    {
        writer.set_formatter("osthread", shepherd_thread_id());
        writer.set_formatter("pikathread", thread_id());
        writer.set_formatter("pikaphase", thread_phase());
        writer.set_formatter("pikaparent", parent_thread_id());
        writer.set_formatter("pikaparentphase", parent_thread_phase());
    }

    void define_formatters_local(logger_writer_type& writer)
    {
        define_common_formatters(writer);
        writer.set_formatter("pikacomponent", dummy_thread_component_id());
    }

    ///////////////////////////////////////////////////////////////////////
    static std::string empty_string;

    log_settings get_log_settings(section const& ini, char const* sec)
    {
        log_settings result;
        if (ini.has_section(sec))
        {
            section const* logini = ini.get_section(sec);
            PIKA_ASSERT(nullptr != logini);

            result.level_ = logini->get_entry("level", empty_string);
            if (!result.level_.empty())
            {
                result.dest_ = logini->get_entry("destination", empty_string);
                result.format_ = unescape(logini->get_entry("format", empty_string));
            }
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////
    // initialize logging for performance measurements
    void init_timing_log(pika::util::logging::level lvl, std::string logdest, std::string logformat,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        if (pika::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = pika::util::timing_logger()->writer();

            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";

            if (logformat.empty()) logformat = "|\\n";

            writer.write(logformat, logdest);
            define_formatters(writer);

            pika::util::timing_logger()->mark_as_initialized();
        }
        pika::util::timing_logger()->set_enabled(lvl);
    }

    void init_timing_log(pika::util::runtime_configuration& ini,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        auto settings = get_log_settings(ini, "pika.logging.timing");

        auto lvl = pika::util::logging::level::disable_all;
        if (!settings.level_.empty())
            lvl = pika::util::detail::get_log_level(settings.level_, true);

        init_timing_log(
            lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_), define_formatters);
    }

    ///////////////////////////////////////////////////////////////////////
    void init_pika_log(pika::util::logging::level lvl, std::string logdest, std::string logformat,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        logger_writer_type& writer = pika::util::pika_logger()->writer();
        logger_writer_type& error_writer = pika::util::pika_error_logger()->writer();

        if (logdest.empty())    // ensure minimal defaults
            logdest = "cerr";

        if (logformat.empty()) logformat = "|\\n";

        if (pika::util::logging::level::disable_all != lvl)
        {
            writer.write(logformat, logdest);
            define_formatters(writer);

            pika::util::pika_logger()->mark_as_initialized();
            pika::util::pika_logger()->set_enabled(lvl);

            // errors are logged to the given destination and to cerr
            if (logdest != "cerr") error_writer.write(logformat, logdest + " cerr");
            define_formatters(error_writer);

            pika::util::pika_error_logger()->mark_as_initialized();
            pika::util::pika_error_logger()->set_enabled(lvl);
        }
        else
        {
            // errors are always logged to cerr
            error_writer.write(logformat, "cerr");

            define_formatters(error_writer);

            pika::util::pika_error_logger()->mark_as_initialized();
            pika::util::pika_error_logger()->set_enabled(pika::util::logging::level::fatal);
        }
    }

    void init_pika_log(pika::util::runtime_configuration& ini,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        auto settings = get_log_settings(ini, "pika.logging");

        auto lvl = pika::util::logging::level::disable_all;
        if (!settings.level_.empty())
            lvl = pika::util::detail::get_log_level(settings.level_, true);

        init_pika_log(
            lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_), define_formatters);
    }

    ///////////////////////////////////////////////////////////////////////
    // initialize logging for application
    void init_app_log(pika::util::logging::level lvl, std::string logdest, std::string logformat,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        if (pika::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = pika::util::app_logger()->writer();

            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";

            if (logformat.empty()) logformat = "|\\n";

            writer.write(logformat, logdest);
            define_formatters(writer);

            pika::util::app_logger()->mark_as_initialized();
        }
        pika::util::app_logger()->set_enabled(lvl);
    }

    void init_app_log(pika::util::runtime_configuration& ini,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        auto settings = get_log_settings(ini, "pika.logging.application");

        auto lvl = pika::util::logging::level::disable_all;
        if (!settings.level_.empty())
            lvl = pika::util::detail::get_log_level(settings.level_, true);

        init_app_log(
            lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_), define_formatters);
    }

    ///////////////////////////////////////////////////////////////////////
    // initialize logging for application
    void init_debuglog_log(pika::util::logging::level lvl, std::string logdest,
        std::string logformat, void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        if (pika::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = pika::util::debuglog_logger()->writer();

            if (logdest.empty())    // ensure minimal defaults
                logdest = "cerr";

            if (logformat.empty()) logformat = "|\\n";

            writer.write(logformat, logdest);
            define_formatters(writer);

            pika::util::debuglog_logger()->mark_as_initialized();
        }
        pika::util::debuglog_logger()->set_enabled(lvl);
    }

    void init_debuglog_log(pika::util::runtime_configuration& ini,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        auto settings = get_log_settings(ini, "pika.logging.debuglog");

        auto lvl = pika::util::logging::level::disable_all;
        if (!settings.level_.empty())
            lvl = pika::util::detail::get_log_level(settings.level_, true);

        init_debuglog_log(
            lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_), define_formatters);
    }

    static void (*default_define_formatters)(
        pika::util::logging::writer::named_write&) = define_formatters_local;

    void init_logging(pika::util::runtime_configuration& ini,
        void (*define_formatters)(pika::util::logging::writer::named_write&))
    {
        default_define_formatters = define_formatters;

        init_timing_log(ini, define_formatters);
        init_pika_log(ini, define_formatters);
        init_app_log(ini, define_formatters);
        init_debuglog_log(ini, define_formatters);
    }

    void init_logging_local(pika::util::runtime_configuration& ini)
    {
        init_logging(ini, define_formatters_local);
    }

    ///////////////////////////////////////////////////////////////////////////
    void disable_logging(logging_destination dest)
    {
        switch (dest)
        {
        case destination_pika:
            pika::util::pika_logger()->set_enabled(pika::util::logging::level::disable_all);
            break;

        case destination_timing:
            pika::util::timing_logger()->set_enabled(pika::util::logging::level::disable_all);
            break;

        case destination_app:
            pika::util::app_logger()->set_enabled(pika::util::logging::level::disable_all);
            break;

        case destination_debuglog:
            pika::util::debuglog_logger()->set_enabled(pika::util::logging::level::disable_all);
            break;
        }
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void enable_logging(logging_destination dest, std::string const& level, std::string logdest,
        std::string logformat)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        auto lvl = pika::util::logging::level::enable_all;
        if (!level.empty()) { lvl = pika::util::detail::get_log_level(level, true); }

        switch (dest)
        {
        case destination_pika:
            detail::init_pika_log(lvl, logdest, logformat, detail::default_define_formatters);
            break;

        case destination_timing:
            detail::init_debuglog_log(lvl, logdest, logformat, detail::default_define_formatters);
            break;

        case destination_app:
            detail::init_app_log(lvl, logdest, logformat, detail::default_define_formatters);
            break;

        case destination_debuglog:
            detail::init_debuglog_log(lvl, logdest, logformat, detail::default_define_formatters);
            break;
        }
    }
}    // namespace pika::detail
