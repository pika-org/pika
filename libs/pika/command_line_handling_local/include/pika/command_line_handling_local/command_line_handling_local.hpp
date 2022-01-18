//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/modules/runtime_configuration.hpp>
#include <pika/modules/util.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pika { namespace local { namespace detail {
    struct command_line_handling
    {
        command_line_handling(pika::util::runtime_configuration rtcfg,
            std::vector<std::string> ini_config,
            util::function_nonser<int(pika::program_options::variables_map& vm)>
                pika_main_f)
          : rtcfg_(rtcfg)
          , ini_config_(ini_config)
          , pika_main_f_(pika_main_f)
          , num_threads_(1)
          , num_cores_(1)
          , pu_step_(1)
          , pu_offset_(std::size_t(-1))
          , numa_sensitive_(0)
          , use_process_mask_(false)
          , cmd_line_parsed_(false)
          , info_printed_(false)
          , version_printed_(false)
        {
        }

        int call(pika::program_options::options_description const& desc_cmdline,
            int argc, char** argv);

        pika::program_options::variables_map vm_;
        pika::util::runtime_configuration rtcfg_;

        std::vector<std::string> ini_config_;
        util::function_nonser<int(pika::program_options::variables_map& vm)>
            pika_main_f_;

        std::size_t num_threads_;
        std::size_t num_cores_;
        std::size_t pu_step_;
        std::size_t pu_offset_;
        std::string queuing_;
        std::string affinity_domain_;
        std::string affinity_bind_;
        std::size_t numa_sensitive_;
        bool use_process_mask_;
        bool cmd_line_parsed_;
        bool info_printed_;
        bool version_printed_;

    protected:
        // Helper functions for checking command line options
        void check_affinity_domain() const;
        void check_affinity_description() const;
        void check_pu_offset() const;
        void check_pu_step() const;

        bool handle_arguments(util::manage_config& cfgmap,
            pika::program_options::variables_map& vm,
            std::vector<std::string>& ini_config);

        void enable_logging_settings(pika::program_options::variables_map& vm,
            std::vector<std::string>& ini_config);

        void store_command_line(int argc, char** argv);
        void store_unregistered_options(std::string const& cmd_name,
            std::vector<std::string> const& unregistered_options);
        bool handle_help_options(
            pika::program_options::options_description const& help);

        void handle_attach_debugger();

        std::vector<std::string> preprocess_config_settings(
            int argc, char** argv);
    };
}}}    // namespace pika::local::detail
