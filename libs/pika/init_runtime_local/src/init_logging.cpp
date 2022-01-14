//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_LOGGING)
#include <pika/init_runtime_local/detail/init_logging.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/runtime_local/get_locality_id.hpp>
#include <pika/runtime_local/get_worker_thread_num.hpp>
#include <pika/threading_base/thread_data.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {

    using logger_writer_type = logging::writer::named_write;

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: shepherd
    struct shepherd_thread_id : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            error_code ec(lightweight);
            std::size_t thread_num = pika::get_worker_thread_num(ec);

            if (std::size_t(-1) != thread_num)
            {
                util::format_to(to, "{:016x}", thread_num);
            }
            else
            {
                to << std::string(16, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix
    struct locality_prefix : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::uint32_t locality_id = pika::get_locality_id();

            if (~static_cast<std::uint32_t>(0) != locality_id)
            {
                util::format_to(to, "{:08x}", locality_id);
            }
            else
            {
                // called from outside a pika thread
                to << std::string(8, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: pika thread id
    struct thread_id : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (nullptr != self)
            {
                // called from inside a pika thread
                threads::thread_id_type id = threads::get_self_id();
                if (id != threads::invalid_thread_id)
                {
                    std::ptrdiff_t value =
                        reinterpret_cast<std::ptrdiff_t>(id.get());
                    util::format_to(to, "{:016x}", value);
                    return;
                }
            }

            // called from outside a pika thread or invalid thread id
            to << std::string(16, '-');
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: pika thread phase
    struct thread_phase : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (nullptr != self)
            {
                // called from inside a pika thread
                std::size_t phase = self->get_thread_phase();
                if (0 != phase)
                {
                    util::format_to(to, "{:04x}", self->get_thread_phase());
                    return;
                }
            }

            // called from outside a pika thread or no phase given
            to << std::string(4, '-');
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix of parent thread
    struct parent_thread_locality : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::uint32_t parent_locality_id =
                threads::get_parent_locality_id();
            if (~static_cast<std::uint32_t>(0) != parent_locality_id)
            {
                // called from inside a pika thread
                util::format_to(to, "{:08x}", parent_locality_id);
            }
            else
            {
                // called from outside a pika thread
                to << std::string(8, '-');
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: pika parent thread id
    struct parent_thread_id : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            threads::thread_id_type parent_id = threads::get_parent_id();
            if (nullptr != parent_id)
            {
                // called from inside a pika thread
                std::ptrdiff_t value =
                    reinterpret_cast<std::ptrdiff_t>(parent_id.get());
                util::format_to(to, "{:016x}", value);
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
    struct parent_thread_phase : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::size_t parent_phase = threads::get_parent_phase();
            if (0 != parent_phase)
            {
                // called from inside a pika thread
                util::format_to(to, "{:04x}", parent_phase);
            }
            else
            {
                // called from outside a pika thread
                to << std::string(4, '-');
            }
        }
    };

#if defined(ANDROID) || defined(__ANDROID__)
    // default log destination for Android
    struct android_log : logging::destination::manipulator
    {
        android_log(char const* tag_)
          : tag(tag_)
        {
        }

        void operator()(logging::message const& msg) override
        {
            __android_log_write(
                ANDROID_LOG_DEBUG, tag.c_str(), msg.full_string().c_str());
        }

        bool operator==(android_log const& rhs) const
        {
            return tag == rhs.tag;
        }

        std::string tag;
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    struct dummy_thread_component_id : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            to << std::string(16, '-');
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    void console_local::operator()(logging::message const& msg)
    {
        switch (dest_)
        {
        default:
        case destination_pika:
            LPIKA_CONSOLE_(level_) << msg;
            break;

        case destination_timing:
            LTIM_CONSOLE_(level_) << msg;
            break;

        case destination_app:
            LAPP_CONSOLE_(level_) << msg;
            break;

        case destination_debuglog:
            LDEB_CONSOLE_ << msg;
            break;
        }
    }

    namespace detail {

        // unescape config entry
        std::string unescape(std::string const& value)
        {
            std::string result;
            std::string::size_type pos = 0;
            std::string::size_type pos1 = value.find_first_of('\\', 0);
            if (std::string::npos != pos1)
            {
                do
                {
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
            writer.set_formatter("locality", locality_prefix());
            writer.set_formatter("pikathread", thread_id());
            writer.set_formatter("pikaphase", thread_phase());
            writer.set_formatter("pikaparent", parent_thread_id());
            writer.set_formatter("pikaparentphase", parent_thread_phase());
            writer.set_formatter("parentloc", parent_thread_locality());
        }

        void define_formatters_local(logger_writer_type& writer)
        {
            define_common_formatters(writer);
            writer.set_formatter("pikacomponent", dummy_thread_component_id());
        }

        ///////////////////////////////////////////////////////////////////////
        static std::string empty_string;

        log_settings get_log_settings(util::section const& ini, char const* sec)
        {
            log_settings result;
            if (ini.has_section(sec))
            {
                util::section const* logini = ini.get_section(sec);
                PIKA_ASSERT(nullptr != logini);

                result.level_ = logini->get_entry("level", empty_string);
                if (!result.level_.empty())
                {
                    result.dest_ =
                        logini->get_entry("destination", empty_string);
                    result.format_ = detail::unescape(
                        logini->get_entry("format", empty_string));
                }
            }
            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        void get_console_local(logger_writer_type& writer, char const* name,
            logging::level lvl, logging_destination dest)
        {
            writer.set_destination(name, console_local(lvl, dest));
        }

        ///////////////////////////////////////////////////////////////////////
        // initialize logging for performance measurements
        void init_timing_log(logging::level lvl, std::string logdest,
            std::string logformat, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            if (pika::util::logging::level::disable_all != lvl)
            {
                logger_writer_type& writer = timing_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
                if (logdest.empty())    // ensure minimal defaults
                    logdest = isconsole ? "android_log" : "console";

                writer.set_destination(
                    "android_log", android_log("pika.timing"));
#else
                if (logdest.empty())    // ensure minimal defaults
                    logdest = isconsole ? "cerr" : "console";
#endif
                if (logformat.empty())
                    logformat = "|\\n";

                set_console_dest(
                    writer, "console", lvl, destination_timing);    //-V106
                writer.write(logformat, logdest);
                define_formatters(writer);

                timing_logger()->mark_as_initialized();
            }
            timing_logger()->set_enabled(lvl);
        }

        void init_timing_log(runtime_configuration& ini, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            auto settings = detail::get_log_settings(ini, "pika.logging.timing");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_timing_log(lvl, PIKA_MOVE(settings.dest_),
                PIKA_MOVE(settings.format_), isconsole, set_console_dest,
                define_formatters);
        }

        ///////////////////////////////////////////////////////////////////////
        void init_pika_log(logging::level lvl, std::string logdest,
            std::string logformat, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            logger_writer_type& writer = pika_logger()->writer();
            logger_writer_type& error_writer = pika_error_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";

            writer.set_destination("android_log", android_log("pika"));
            error_writer.set_destination("android_log", android_log("pika"));
#else
            if (logdest.empty())    // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            if (pika::util::logging::level::disable_all != lvl)
            {
                set_console_dest(
                    writer, "console", lvl, destination_pika);    //-V106
                writer.write(logformat, logdest);
                define_formatters(writer);

                pika_logger()->mark_as_initialized();
                pika_logger()->set_enabled(lvl);

                // errors are logged to the given destination and to cerr
                set_console_dest(
                    error_writer, "console", lvl, destination_pika);    //-V106
#if !defined(ANDROID) && !defined(__ANDROID__)
                if (logdest != "cerr")
                    error_writer.write(logformat, logdest + " cerr");
#endif
                define_formatters(error_writer);

                pika_error_logger()->mark_as_initialized();
                pika_error_logger()->set_enabled(lvl);
            }
            else
            {
                // errors are always logged to cerr
                if (!isconsole)
                {
                    set_console_dest(
                        writer, "console", lvl, destination_pika);    //-V106
                    error_writer.write(logformat, "console");
                }
                else
                {
#if defined(ANDROID) || defined(__ANDROID__)
                    error_writer.write(logformat, "android_log");
#else
                    error_writer.write(logformat, "cerr");
#endif
                }
                define_formatters(error_writer);

                pika_error_logger()->mark_as_initialized();
                pika_error_logger()->set_enabled(
                    pika::util::logging::level::fatal);
            }
        }

        void init_pika_log(runtime_configuration& ini, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            auto settings = detail::get_log_settings(ini, "pika.logging");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_pika_log(lvl, PIKA_MOVE(settings.dest_),
                PIKA_MOVE(settings.format_), isconsole, set_console_dest,
                define_formatters);
        }

        ///////////////////////////////////////////////////////////////////////
        // initialize logging for application
        void init_app_log(logging::level lvl, std::string logdest,
            std::string logformat, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            if (pika::util::logging::level::disable_all != lvl)
            {
                logger_writer_type& writer = app_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
                if (logdest.empty())    // ensure minimal defaults
                    logdest = isconsole ? "android_log" : "console";
                writer.set_destination(
                    "android_log", android_log("pika.application"));
#else
                if (logdest.empty())    // ensure minimal defaults
                    logdest = isconsole ? "cerr" : "console";
#endif
                if (logformat.empty())
                    logformat = "|\\n";

                set_console_dest(
                    writer, "console", lvl, destination_app);    //-V106
                writer.write(logformat, logdest);
                define_formatters(writer);

                app_logger()->mark_as_initialized();
            }
            app_logger()->set_enabled(lvl);
        }

        void init_app_log(runtime_configuration& ini, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            auto settings =
                detail::get_log_settings(ini, "pika.logging.application");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_app_log(lvl, PIKA_MOVE(settings.dest_),
                PIKA_MOVE(settings.format_), isconsole, set_console_dest,
                define_formatters);
        }

        ///////////////////////////////////////////////////////////////////////
        // initialize logging for application
        void init_debuglog_log(logging::level lvl, std::string logdest,
            std::string logformat, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            if (pika::util::logging::level::disable_all != lvl)
            {
                logger_writer_type& writer = debuglog_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
                if (logdest.empty())    // ensure minimal defaults
                    logdest = isconsole ? "android_log" : "console";
                writer.set_destination(
                    "android_log", android_log("pika.debuglog"));
#else
                if (logdest.empty())    // ensure minimal defaults
                    logdest = isconsole ? "cerr" : "console";
#endif
                if (logformat.empty())
                    logformat = "|\\n";

                set_console_dest(
                    writer, "console", lvl, destination_debuglog);    //-V106
                writer.write(logformat, logdest);
                define_formatters(writer);

                debuglog_logger()->mark_as_initialized();
            }
            debuglog_logger()->set_enabled(lvl);
        }

        void init_debuglog_log(runtime_configuration& ini, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            auto settings =
                detail::get_log_settings(ini, "pika.logging.debuglog");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_debuglog_log(lvl, PIKA_MOVE(settings.dest_),
                PIKA_MOVE(settings.format_), isconsole, set_console_dest,
                define_formatters);
        }

        ///////////////////////////////////////////////////////////////////////
        void init_timing_console_log(
            logging::level lvl, std::string logdest, std::string logformat)
        {
            if (pika::util::logging::level::disable_all != lvl)
            {
                logger_writer_type& writer = timing_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "android_log";
                writer.set_destination(
                    "android_log", android_log("pika.timing"));
#else
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "cerr";
#endif
                if (logformat.empty())
                    logformat = "|\\n";

                writer.write(logformat, logdest);

                timing_console_logger()->mark_as_initialized();
            }
            timing_console_logger()->set_enabled(lvl);
        }

        void init_timing_console_log(util::section const& ini)
        {
            auto settings =
                detail::get_log_settings(ini, "pika.logging.console.timing");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_timing_console_log(
                lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_));
        }

        ///////////////////////////////////////////////////////////////////////
        void init_pika_console_log(
            logging::level lvl, std::string logdest, std::string logformat)
        {
            if (pika::util::logging::level::disable_all != lvl)
            {
                logger_writer_type& writer = pika_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "android_log";
                writer.set_destination("android_log", android_log("pika"));
#else
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "cerr";
#endif
                if (logformat.empty())
                    logformat = "|\\n";

                writer.write(logformat, logdest);

                pika_console_logger()->mark_as_initialized();
            }
            pika_console_logger()->set_enabled(lvl);
        }

        void init_pika_console_log(util::section const& ini)
        {
            auto settings =
                detail::get_log_settings(ini, "pika.logging.console");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_pika_console_log(
                lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_));
        }

        ///////////////////////////////////////////////////////////////////////
        void init_app_console_log(
            logging::level lvl, std::string logdest, std::string logformat)
        {
            if (pika::util::logging::level::disable_all != lvl)
            {
                logger_writer_type& writer = app_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "android_log";
                writer.set_destination(
                    "android_log", android_log("pika.application"));
#else
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "cerr";
#endif
                if (logformat.empty())
                    logformat = "|\\n";

                writer.write(logformat, logdest);

                app_console_logger()->mark_as_initialized();
            }
            app_console_logger()->set_enabled(lvl);
        }

        void init_app_console_log(util::section const& ini)
        {
            auto settings = detail::get_log_settings(
                ini, "pika.logging.console.application");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_app_console_log(
                lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_));
        }

        ///////////////////////////////////////////////////////////////////////
        void init_debuglog_console_log(
            logging::level lvl, std::string logdest, std::string logformat)
        {
            if (pika::util::logging::level::disable_all != lvl)
            {
                logger_writer_type& writer =
                    debuglog_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "android_log";
                writer.set_destination(
                    "android_log", android_log("pika.debuglog"));
#else
                if (logdest.empty())    // ensure minimal defaults
                    logdest = "cerr";
#endif
                if (logformat.empty())
                    logformat = "|\\n";

                writer.write(logformat, logdest);

                debuglog_console_logger()->mark_as_initialized();
            }
            debuglog_console_logger()->set_enabled(lvl);
        }

        void init_debuglog_console_log(util::section const& ini)
        {
            auto settings =
                detail::get_log_settings(ini, "pika.logging.console.debuglog");

            auto lvl = pika::util::logging::level::disable_all;
            if (!settings.level_.empty())
                lvl = detail::get_log_level(settings.level_, true);

            init_debuglog_console_log(
                lvl, PIKA_MOVE(settings.dest_), PIKA_MOVE(settings.format_));
        }

        ///////////////////////////////////////////////////////////////////////
        static void (*default_set_console_dest)(logger_writer_type&,
            char const*, logging::level,
            logging_destination) = get_console_local;

        static void (*default_define_formatters)(
            logging::writer::named_write&) = define_formatters_local;

        static bool default_isconsole = true;

        void init_logging(runtime_configuration& ini, bool isconsole,
            void (*set_console_dest)(logger_writer_type&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&))
        {
            default_isconsole = isconsole;
            default_set_console_dest = set_console_dest;
            default_define_formatters = define_formatters;

            // initialize normal logs
            init_timing_log(
                ini, isconsole, set_console_dest, define_formatters);
            init_pika_log(ini, isconsole, set_console_dest, define_formatters);
            init_app_log(ini, isconsole, set_console_dest, define_formatters);
            init_debuglog_log(
                ini, isconsole, set_console_dest, define_formatters);

            // initialize console logs
            init_timing_console_log(ini);
            init_pika_console_log(ini);
            init_app_console_log(ini);
            init_debuglog_console_log(ini);
        }

        void init_logging_local(runtime_configuration& ini)
        {
            init_logging(ini, true, util::detail::get_console_local,
                util::detail::define_formatters_local);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    void disable_logging(logging_destination dest)
    {
        switch (dest)
        {
        case destination_pika:
            pika_logger()->set_enabled(logging::level::disable_all);
            pika_console_logger()->set_enabled(logging::level::disable_all);
            break;

        case destination_timing:
            timing_logger()->set_enabled(logging::level::disable_all);
            timing_console_logger()->set_enabled(logging::level::disable_all);
            break;

        case destination_app:
            app_logger()->set_enabled(logging::level::disable_all);
            app_console_logger()->set_enabled(logging::level::disable_all);
            break;

        case destination_debuglog:
            debuglog_logger()->set_enabled(logging::level::disable_all);
            debuglog_console_logger()->set_enabled(logging::level::disable_all);
            break;
        }
    }

    void enable_logging(logging_destination dest, std::string const& level,
        std::string logdest, std::string logformat)
    {
        auto lvl = pika::util::logging::level::enable_all;
        if (!level.empty())
        {
            lvl = detail::get_log_level(level, true);
        }

        switch (dest)
        {
        case destination_pika:
            detail::init_pika_log(lvl, logdest, logformat,
                detail::default_isconsole, detail::default_set_console_dest,
                detail::default_define_formatters);
            detail::init_pika_console_log(
                lvl, PIKA_MOVE(logdest), PIKA_MOVE(logformat));
            break;

        case destination_timing:
            detail::init_debuglog_log(lvl, logdest, logformat,
                detail::default_isconsole, detail::default_set_console_dest,
                detail::default_define_formatters);
            detail::init_debuglog_console_log(
                lvl, PIKA_MOVE(logdest), PIKA_MOVE(logformat));
            break;

        case destination_app:
            detail::init_app_log(lvl, logdest, logformat,
                detail::default_isconsole, detail::default_set_console_dest,
                detail::default_define_formatters);
            detail::init_app_console_log(
                lvl, PIKA_MOVE(logdest), PIKA_MOVE(logformat));
            break;

        case destination_debuglog:
            detail::init_debuglog_log(lvl, logdest, logformat,
                detail::default_isconsole, detail::default_set_console_dest,
                detail::default_define_formatters);
            detail::init_debuglog_console_log(
                lvl, PIKA_MOVE(logdest), PIKA_MOVE(logformat));
            break;
        }
    }
}}    // namespace pika::util

#else

#include <pika/init_runtime_local/detail/init_logging.hpp>
#include <pika/modules/logging.hpp>
#include <pika/util/get_entry_as.hpp>

#include <iostream>
#include <string>

namespace pika { namespace util {

    //////////////////////////////////////////////////////////////////////////
    void enable_logging(
        logging_destination, std::string const&, std::string, std::string)
    {
    }

    void disable_logging(logging_destination) {}

    //////////////////////////////////////////////////////////////////////////
    namespace detail {

        void warn_if_logging_requested(runtime_configuration& ini)
        {
            using util::get_entry_as;

            // warn if logging is requested
            if (get_entry_as<int>(ini, "pika.logging.level", -1) > 0 ||
                get_entry_as<int>(ini, "pika.logging.timing.level", -1) > 0 ||
                get_entry_as<int>(ini, "pika.logging.debuglog.level", -1) > 0 ||
                get_entry_as<int>(ini, "pika.logging.application.level", -1) > 0)
            {
                std::cerr
                    << "pika::init_logging: warning: logging is requested even "
                       "while it was disabled at compile time. If you "
                       "need logging to be functional, please reconfigure and "
                       "rebuild pika with PIKA_WITH_LOGGING set to ON."
                    << std::endl;
            }
        }
    }    // namespace detail
}}       // namespace pika::util

#endif    // PIKA_HAVE_LOGGING
