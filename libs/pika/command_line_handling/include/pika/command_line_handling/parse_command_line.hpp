//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/ini/ini.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/variables_map.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace pika::detail {
    enum class commandline_error_mode
    {
        return_on_error,
        rethrow_on_error,
        allow_unregistered,
        ignore_aliases = 0x40,
        report_missing_config_file = 0x80
    };

    commandline_error_mode operator&(commandline_error_mode const lhs,
        commandline_error_mode const rhs) noexcept;
    commandline_error_mode& operator&=(
        commandline_error_mode& lhs, commandline_error_mode rhs) noexcept;
    commandline_error_mode operator|(
        commandline_error_mode const lhs, commandline_error_mode rhs) noexcept;
    commandline_error_mode& operator|=(
        commandline_error_mode& lhs, commandline_error_mode rhs) noexcept;
    commandline_error_mode operator~(commandline_error_mode m) noexcept;
    bool contains_error_mode(commandline_error_mode const m,
        commandline_error_mode const b) noexcept;
    std::string enquote(std::string const& arg);

    bool parse_commandline(pika::util::section const& rtcfg,
        pika::program_options::options_description const& app_options,
        std::string const& cmdline, pika::program_options::variables_map& vm,
        commandline_error_mode error_mode =
            commandline_error_mode::return_on_error,
        pika::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);

    bool parse_commandline(pika::util::section const& rtcfg,
        pika::program_options::options_description const& app_options,
        std::string const& arg0, std::vector<std::string> const& args,
        pika::program_options::variables_map& vm,
        commandline_error_mode error_mode =
            commandline_error_mode::return_on_error,
        pika::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);

    std::string reconstruct_command_line(
        pika::program_options::variables_map const& vm);
}    // namespace pika::detail
