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

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <ctime>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace pika::detail {
    static void spdlog_format_thread_id(pika::threads::detail::thread_id_type const id,
        const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest)
    {
        if (id)
        {
            dest.append(fmt::format("{}/{}", id,
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
                id ? get_thread_id_data(id)->get_description() :
#endif
                     "----"));
        }
        else { dest.append(std::string_view("----/----")); }
    }

    class pika_thread_id_formatter_flag : public spdlog::custom_flag_formatter
    {
    public:
        void format(const spdlog::details::log_msg& m, const std::tm& t,
            spdlog::memory_buf_t& dest) override
        {
            spdlog_format_thread_id(threads::detail::get_self_id(), m, t, dest);
        }

        std::unique_ptr<custom_flag_formatter> clone() const override
        {
            return spdlog::details::make_unique<pika_thread_id_formatter_flag>();
        }
    };

    class pika_parent_thread_id_formatter_flag : public spdlog::custom_flag_formatter
    {
    public:
        void format(const spdlog::details::log_msg& m, const std::tm& t,
            spdlog::memory_buf_t& dest) override
        {
            spdlog_format_thread_id(threads::detail::get_parent_id(), m, t, dest);
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
            if (i != std::size_t(-1)) { dest.append(fmt::format("{:04}", i)); }
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
            static PIKA_DETAIL_NS_DEBUG::hostname_print_helper helper{};
            static std::string_view hostname_str = helper.get_hostname();
            dest.append(hostname_str);

            if (int rank = helper.guess_rank(); rank != -1)
            {
                dest.append(hostname_str);
                dest.append(fmt::format("/{}", rank));
            }
            else { dest.append(std::string_view("/----")); }
        }

        std::unique_ptr<custom_flag_formatter> clone() const override
        {
            return spdlog::details::make_unique<hostname_formatter_flag>();
        }
    };

    struct log_settings
    {
        std::string level_;
        std::string dest_;
        std::string format_;
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
        auto& sinks = get_pika_logger().sinks();
        sinks.clear();
        sinks.push_back(get_spdlog_sink(settings.dest_));

        // Set log pattern
        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<pika_thread_id_formatter_flag>('k');
        formatter->add_flag<pika_parent_thread_id_formatter_flag>('q');
        formatter->add_flag<pika_worker_thread_formatter_flag>('w');
        formatter->add_flag<hostname_formatter_flag>('j');
        formatter->set_pattern(settings.format_);
        get_pika_logger().set_formatter(std::move(formatter));

        // Set log level
        get_pika_logger().set_level(get_spdlog_level(settings.level_));
    }
}    // namespace pika::detail
