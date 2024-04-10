//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>

#include <pika/assert.hpp>
#include <pika/debugging/print.hpp>
#include <pika/init_runtime/detail/init_logging.hpp>
#include <pika/runtime/get_worker_thread_num.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/string_util/from_string.hpp>
#include <pika/threading_base/thread_data.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace pika::detail {
    class pika_thread_id_formatter_flag : public spdlog::custom_flag_formatter
    {
    public:
        void format(
            const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override
        {
            const auto id = threads::detail::get_self_id();
            if (id != threads::detail::invalid_thread_id)
            {
                dest.append(fmt::format("{:016x}", reinterpret_cast<std::ptrdiff_t>(id.get())));
            }
            else { dest.append(std::string_view("----")); }
        }

        std::unique_ptr<custom_flag_formatter> clone() const override
        {
            return spdlog::details::make_unique<pika_thread_id_formatter_flag>();
        }
    };

    class pika_parent_thread_id_formatter_flag : public spdlog::custom_flag_formatter
    {
    public:
        void format(
            const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override
        {
            auto id = threads::detail::get_parent_id();
            if (id != nullptr)
            {
                dest.append(fmt::format("{:016x}", reinterpret_cast<std::ptrdiff_t>(id.get())));
            }
            else { dest.append(std::string_view("----")); }
        }

        std::unique_ptr<custom_flag_formatter> clone() const override
        {
            return spdlog::details::make_unique<pika_parent_thread_id_formatter_flag>();
        }
    };

    class pika_worker_thread_formatter_flag : public spdlog::custom_flag_formatter
    {
        static void format_id(spdlog::memory_buf_t& dest, std::size_t i)
        {
            if (i != std::size_t(-1)) { dest.append(fmt::format("{:04x}", i)); }
            else { dest.append(std::string_view("----")); }
        }

    public:
        void format(
            const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override
        {
            format_id(dest, pika::get_thread_pool_num());
            dest.append(std::string_view("/"));
            format_id(dest, pika::get_worker_thread_num());
            dest.append(std::string_view("/"));
            format_id(dest, pika::get_local_worker_thread_num());
        }

        std::unique_ptr<custom_flag_formatter> clone() const override
        {
            return spdlog::details::make_unique<pika_worker_thread_formatter_flag>();
        }
    };

    class hostname_formatter_flag : public spdlog::custom_flag_formatter
    {
    public:
        void format(
            const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override
        {
            static std::string_view hostname_str =
                PIKA_DETAIL_NS_DEBUG::hostname_print_helper{}.get_hostname();
            dest.append(hostname_str);
        }

        std::unique_ptr<custom_flag_formatter> clone() const override
        {
            return spdlog::details::make_unique<hostname_formatter_flag>();
        }
    };

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

    void define_formatters(logger_writer_type& writer)
    {
        writer.set_formatter("osthread", shepherd_thread_id());
        writer.set_formatter("pikathread", thread_id());
        writer.set_formatter("pikaphase", thread_phase());
        writer.set_formatter("pikaparent", parent_thread_id());
        writer.set_formatter("pikaparentphase", parent_thread_phase());
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
    void init_pika_log(pika::util::logging::level lvl, std::string logdest, std::string logformat)
    {
        // TODO
    }

    void init_pika_log(pika::util::runtime_configuration& ini)
    {
        // TODO
    }

    void init_logging(pika::util::runtime_configuration& ini)
    {
        init_pika_log(ini);

        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<pika_thread_id_formatter_flag>('k');           // TODO: What letter?
        formatter->add_flag<pika_parent_thread_id_formatter_flag>('q');    // TODO: What letter?
        formatter->add_flag<pika_worker_thread_formatter_flag>('w');       // TODO: What letter?
        formatter->add_flag<hostname_formatter_flag>('j');                 // TODO: What letter?
        formatter->set_pattern("[%Y-%m-%d %H:%M:%S.%F] [%n] [%-8l] [host:%j] [pid:%P] [tid:%t] "
                               "[pool:%w] [parent:%q] [task:%k] [%s:%#:%!] %v");
        pika::util::get_pika_logger()->set_formatter(std::move(formatter));

        auto settings = get_log_settings(ini, "pika.logging");
        auto lvl = spdlog::level::off;
        std::cerr << "settings.level_: " << settings.level_ << "\n";
        if (!settings.level_.empty())
        {
            lvl = pika::util::detail::get_spdlog_level(settings.level_);
        }
        pika::util::get_pika_logger()->set_level(lvl);
    }

    ///////////////////////////////////////////////////////////////////////////
    void disable_logging(logging_destination dest)
    {
        switch (dest)
        {
        case destination_pika:
            // TODO
            // pika::util::pika_logger()->set_enabled(pika::util::logging::level::disable_all);
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
        case destination_pika: /* TODO detail::init_pika_log(lvl, logdest, logformat); */ break;
        }
    }
}    // namespace pika::detail
