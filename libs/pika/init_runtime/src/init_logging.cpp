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
            dest.append(fmt::format("{}", threads::detail::get_self_id()));
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
            dest.append(fmt::format("{}", threads::detail::get_parent_id()));
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

    static log_settings get_log_settings(section const& ini, char const* sec)
    {
        log_settings result;
        if (ini.has_section(sec))
        {
            section const* logini = ini.get_section(sec);
            PIKA_ASSERT(nullptr != logini);

            result.level_ = logini->get_entry("level", "");
            result.dest_ = logini->get_entry("destination", "");
            result.format_ = logini->get_entry("format", "");
        }
        return result;
    }

    void init_logging(pika::util::runtime_configuration& ini)
    {
        auto settings = get_log_settings(ini, "pika.log");

        // Set log destination
        auto& sinks = get_pika_logger()->sinks();
        sinks.clear();
        sinks.push_back(get_spdlog_sink(settings.dest_));

        // Set log pattern
        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<pika_thread_id_formatter_flag>('k');
        formatter->add_flag<pika_parent_thread_id_formatter_flag>('q');
        formatter->add_flag<pika_worker_thread_formatter_flag>('w');
        formatter->add_flag<hostname_formatter_flag>('j');
        formatter->set_pattern(settings.format_);
        get_pika_logger()->set_formatter(std::move(formatter));

        // Set log level
        get_pika_logger()->set_level(get_spdlog_level(settings.level_));
    }
}    // namespace pika::detail
