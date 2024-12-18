//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/command_line_handling/parse_command_line.hpp>
#include <pika/detail/filesystem.hpp>
#include <pika/ini/ini.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/parsers.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/string_util/from_string.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <any>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika::detail {
    commandline_error_mode operator&(
        commandline_error_mode const lhs, commandline_error_mode const rhs) noexcept
    {
        return static_cast<commandline_error_mode>(
            static_cast<std::underlying_type_t<commandline_error_mode>>(lhs) &
            static_cast<std::underlying_type_t<commandline_error_mode>>(rhs));
    }

    commandline_error_mode& operator&=(
        commandline_error_mode& lhs, commandline_error_mode rhs) noexcept
    {
        lhs = static_cast<commandline_error_mode>(
            static_cast<std::underlying_type_t<commandline_error_mode>>(lhs) &
            static_cast<std::underlying_type_t<commandline_error_mode>>(rhs));
        return lhs;
    }

    commandline_error_mode operator|(
        commandline_error_mode const lhs, commandline_error_mode rhs) noexcept
    {
        return static_cast<commandline_error_mode>(
            static_cast<std::underlying_type_t<commandline_error_mode>>(lhs) |
            static_cast<std::underlying_type_t<commandline_error_mode>>(rhs));
    }

    commandline_error_mode& operator|=(
        commandline_error_mode& lhs, commandline_error_mode rhs) noexcept
    {
        lhs = static_cast<commandline_error_mode>(
            static_cast<std::underlying_type_t<commandline_error_mode>>(lhs) |
            static_cast<std::underlying_type_t<commandline_error_mode>>(rhs));
        return lhs;
    }

    commandline_error_mode operator~(commandline_error_mode m) noexcept
    {
        return static_cast<commandline_error_mode>(
            ~static_cast<std::underlying_type_t<commandline_error_mode>>(m));
    }

    bool contains_error_mode(
        commandline_error_mode const m, commandline_error_mode const b) noexcept
    {
        return static_cast<std::underlying_type_t<commandline_error_mode>>(m) &
            static_cast<std::underlying_type_t<commandline_error_mode>>(b);
    }

    std::string enquote(std::string const& arg)
    {
        if (arg.find_first_of(" \t\"") != std::string::npos) return std::string("\"") + arg + "\"";
        return arg;
    }

    static std::string trim_whitespace(std::string const& s)
    {
        using size_type = std::string::size_type;

        size_type first = s.find_first_not_of(" \t");
        if (std::string::npos == first) return std::string();

        size_type last = s.find_last_not_of(" \t");
        return s.substr(first, last - first + 1);
    }

    ///////////////////////////////////////////////////////////////////////
    // Additional command line parser which interprets '@something' as an
    // option "options-file" with the value "something". Additionally we
    // resolve defined command line option aliases.
    struct option_parser
    {
        option_parser(pika::detail::section const& ini, bool ignore_aliases)
          : ini_(ini)
          , ignore_aliases_(ignore_aliases)
        {
        }

        std::pair<std::string, std::string> operator()(std::string const& s) const
        {
            // handle special syntax for configuration files @filename
            if ('@' == s[0]) return std::make_pair(std::string("pika:options-file"), s.substr(1));

            return std::make_pair(std::string(), std::string());
        }

        pika::detail::section const& ini_;
        bool ignore_aliases_;
    };

    ///////////////////////////////////////////////////////////////////////
    pika::program_options::basic_command_line_parser<char>& get_commandline_parser(
        pika::program_options::basic_command_line_parser<char>& p, commandline_error_mode mode)
    {
        if ((mode & ~commandline_error_mode::report_missing_config_file) ==
            commandline_error_mode::allow_unregistered)
            return p.allow_unregistered();
        return p;
    }

    ///////////////////////////////////////////////////////////////////////
    // Read all options from a given config file, parse and add them to the
    // given variables_map
    bool read_config_file_options(std::string const& filename,
        pika::program_options::options_description const& desc,
        pika::program_options::variables_map& vm, pika::detail::section const& rtcfg,
        commandline_error_mode error_mode)
    {
        std::ifstream ifs(filename.c_str());
        if (!ifs.is_open())
        {
            if (contains_error_mode(error_mode, commandline_error_mode::report_missing_config_file))
            {
                std::cerr
                    << "pika::init: command line warning: command line options file not found ("
                    << filename << ")" << std::endl;
            }
            return false;
        }

        std::vector<std::string> options;
        std::string line;
        while (std::getline(ifs, line))
        {
            // skip empty lines
            std::string::size_type pos = line.find_first_not_of(" \t");
            if (pos == std::string::npos) continue;

            // strip leading and trailing whitespace
            line = trim_whitespace(line);

            // skip comment lines
            if ('#' != line[0])
            {
                std::string::size_type p1 = line.find_first_of(" \t");
                if (p1 != std::string::npos)
                {
                    // rebuild the line connecting the parts with a '='
                    line = trim_whitespace(line.substr(0, p1)) + '=' +
                        trim_whitespace(line.substr(p1));
                }
                options.push_back(line);
            }
        }

        // add options to parsed settings
        if (!options.empty())
        {
            using pika::program_options::basic_command_line_parser;
            using pika::program_options::command_line_parser;
            using pika::program_options::store;
            using pika::program_options::value;
            using namespace pika::program_options::command_line_style;

            store(detail::get_commandline_parser(
                      command_line_parser(options)
                          .options(desc)
                          .style(unix_style)
                          .extra_parser(detail::option_parser(rtcfg,
                              contains_error_mode(
                                  error_mode, commandline_error_mode::ignore_aliases))),
                      error_mode & ~commandline_error_mode::ignore_aliases)
                      .run(),
                vm);
            notify(vm);
        }
        return true;
    }

    // try to find a config file somewhere up the filesystem hierarchy
    // starting with the input file path. This allows to use a general
    // <app_name>.cfg file for all executables in a certain project.
    void handle_generic_config_options(std::string appname,
        pika::program_options::variables_map& vm,
        pika::program_options::options_description const& desc_cfgfile,
        pika::detail::section const& ini, commandline_error_mode error_mode)
    {
        if (appname.empty()) return;

        std::filesystem::path dir(filesystem::initial_path());
        std::filesystem::path app(appname);
        appname = pika::detail::filesystem::basename(app.filename());

        // walk up the hierarchy, trying to find a file <appname>.cfg
        while (!dir.empty())
        {
            std::filesystem::path filename = dir / (appname + ".cfg");
            bool result = read_config_file_options(filename.string(), desc_cfgfile, vm, ini,
                error_mode & ~commandline_error_mode::report_missing_config_file);
            if (result) break;    // break on the first options file found

            auto dir_prev = dir;
            dir = dir.parent_path();    // chop off last directory part
            if (dir_prev == dir) break;
        }
    }

    // handle all --options-config found on the command line
    void handle_config_options(pika::program_options::variables_map& vm,
        pika::program_options::options_description const& desc_cfgfile,
        pika::detail::section const& ini, commandline_error_mode error_mode)
    {
        using pika::program_options::options_description;
        if (vm.count("pika:options-file"))
        {
            std::vector<std::string> const& cfg_files =
                vm["pika:options-file"].as<std::vector<std::string>>();

            for (std::string const& cfg_file : cfg_files)
            {
                // parse a single config file and store the results
                read_config_file_options(cfg_file, desc_cfgfile, vm, ini, error_mode);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    void parse_commandline(pika::detail::section const& rtcfg,
        pika::program_options::options_description const& app_options, std::string const& arg0,
        std::vector<std::string> const& args, pika::program_options::variables_map& vm,
        commandline_error_mode error_mode, pika::program_options::options_description* visible,
        std::vector<std::string>* unregistered_options)
    {
        using pika::program_options::basic_command_line_parser;
        using pika::program_options::command_line_parser;
        using pika::program_options::options_description;
        using pika::program_options::parsed_options;
        using pika::program_options::positional_options_description;
        using pika::program_options::store;
        using pika::program_options::value;
        using namespace pika::program_options::command_line_style;

        // clang-format off
        options_description cmdline_options(
            "pika options (allowed on command line only)");
        cmdline_options.add_options()
            ("pika:help", "print out program usage (this message)")
            ("pika:version", "print out pika version and copyright information")
            ("pika:info", "print out pika configuration information")
            ("pika:options-file", value<std::vector<std::string> >()->composing(),
                "specify a file containing command line options "
                "(alternatively: @filepath)")
        ;

        options_description pika_options(
            "pika options (additionally allowed in an options file)");
        options_description hidden_options("Hidden options");

        // general options definitions
        pika_options.add_options()
            ("pika:pu-offset", value<std::size_t>(),
                "the first processing unit this instance of pika should be "
                "run on (default: 0), valid for "
                "--pika:queuing=local, --pika:queuing=abp-priority, "
                "--pika:queuing=static, --pika:queuing=static-priority, "
                "and --pika:queuing=local-priority only")
            ("pika:pu-step", value<std::size_t>(),
                "the step between used processing unit numbers for this "
                "instance of pika (default: 1), valid for "
                "--pika:queuing=local, --pika:queuing=abp-priority, "
                "--pika:queuing=static, --pika:queuing=static-priority "
                "and --pika:queuing=local-priority only")
            ("pika:affinity", value<std::string>(),
                "the affinity domain the OS threads will be confined to, "
                "possible values: pu, core, numa, machine (default: pu), valid for "
                "--pika:queuing=local, --pika:queuing=abp-priority, "
                "--pika:queuing=static, --pika:queuing=static-priority "
                " and --pika:queuing=local-priority only")
            ("pika:bind", value<std::vector<std::string> >()->composing(),
                "the detailed affinity description for the OS threads, see "
                "the documentation for a detailed description of possible "
                "values. Do not use with --pika:pu-step, --pika:pu-offset, or "
                "--pika:affinity options. Implies --pika:numa-sensitive=1"
                "(--pika:bind=none disables defining thread affinities).")
            ("pika:process-mask", value<std::string>(),
                "a process mask in hexadecimal form to restrict cores available for "
                "the pika runtime. If a mask has been set externally on the executable, "
                "this option overrides that mask. Has no effect if "
                "--pika:ignore-process-mask is used. This option does not set the "
                "process mask for the main thread. The mask only affects threads spawned "
                "by the pika runtime.")
            ("pika:ignore-process-mask", "ignore the process mask to restrict "
                "available hardware resources, use all available processing units")
            ("pika:print-bind",
                "print to the console the bit masks calculated from the "
                "arguments specified to all --pika:bind options.")
            ("pika:threads", value<std::string>(),
                "the number of operating system threads to spawn for the pika "
                "runtime (default: cores, using 'all' will spawn one thread for "
                "each processing unit")
            ("pika:cores", value<std::string>(),
                "the number of cores to utilize for the pika "
                "runtime (default: 'all', i.e. the number of cores is based on "
                "the number of total cores in the system)")
            ("pika:queuing", value<std::string>(),
                "the queue scheduling policy to use, options are "
                "'local', 'local-priority-fifo','local-priority-lifo', "
                "'abp-priority-fifo', 'abp-priority-lifo', 'static', and "
                "'static-priority' (default: 'local-priority'; "
                "all option values can be abbreviated)")
            ("pika:high-priority-threads", value<std::size_t>(),
                "the number of operating system threads maintaining a high "
                "priority queue (default: number of OS threads), valid for "
                "--pika:queuing=local-priority,--pika:queuing=static-priority, "
                " and --pika:queuing=abp-priority only)")
            ("pika:numa-sensitive", value<std::size_t>()->implicit_value(0),
                "makes the local-priority scheduler NUMA sensitive ("
                "allowed values: 0 - no NUMA sensitivity, 1 - allow only for "
                "boundary cores to steal across NUMA domains, 2 - "
                "no cross boundary stealing is allowed (default value: 0)")
#if defined(PIKA_HAVE_MPI)
            ("pika:mpi-completion-mode", value<std::size_t>(),
                "the pika MPI polling completion/handler mode "
                "(only available if MPI built with MPI support)")
            ("pika:mpi-enable-pool",
                "enable the creation of a dedicated pool for mpi-polling "
                "(only available if MPI built with MPI support and available threads>1).")
#endif
        ;

        options_description config_options("pika configuration options");
        config_options.add_options()
            ("pika:app-config", value<std::string>(),
                "load the specified application configuration (ini) file")
            ("pika:config", value<std::string>()->default_value(""),
                "load the specified pika configuration (ini) file")
            ("pika:ini", value<std::vector<std::string> >()->composing(),
                "add a configuration definition to the default runtime "
                "configuration")
            ("pika:exit", "exit after configuring the runtime")
        ;

        static std::string log_level_description = fmt::format(
            "set log level, allowed values are {} (trace) to {} (off) (default: {} (warn))",
            SPDLOG_LEVEL_TRACE, SPDLOG_LEVEL_OFF, SPDLOG_LEVEL_WARN);

        options_description debugging_options("pika debugging options");
        debugging_options.add_options()
            ("pika:dump-config-initial", "print the initial runtime configuration")
            ("pika:dump-config", "print the final runtime configuration")
            // enable debug output from command line handling
            ("pika:debug-clp", "debug command line processing")
            ("pika:log-destination", value<std::string>(),
              "set log destination (default: cerr)")
            ("pika:log-level", value<std::underlying_type_t<spdlog::level::level_enum>>(),
                log_level_description.c_str())
            ("pika:log-format", value<std::string>(), "set log format string")
#if defined(_POSIX_VERSION) || defined(PIKA_WINDOWS)
            ("pika:attach-debugger",
                value<std::string>()->implicit_value("startup"),
                "wait for a debugger to be attached, possible values: "
                "off, startup, exception or test-failure (default: startup)")
#endif
        ;

        hidden_options.add_options()
            ("pika:ignore", "this option will be silently ignored")
        ;

        // construct the overall options description and parse the
        // command line
        options_description desc_cmdline;
        options_description positional_options;
        desc_cmdline
            .add(app_options).add(cmdline_options)
            .add(pika_options)
            .add(config_options).add(debugging_options)
            .add(hidden_options)
        ;

        options_description desc_cfgfile;
        desc_cfgfile
            .add(app_options).add(pika_options)
            .add(config_options)
            .add(debugging_options).add(hidden_options)
        ;

        if (rtcfg.get_entry("pika.commandline.allow_unknown", "0") == "0")
        {
            // move all positional options into the pika:positional option
            // group
            positional_options_description pd;
            pd.add("pika:positional", -1);

            positional_options.add_options()
                ("pika:positional", value<std::vector<std::string> >(),
                    "positional options")
            ;
            desc_cmdline.add(positional_options);
            desc_cfgfile.add(positional_options);

            // parse command line, allow for unregistered options this point
            parsed_options opts(detail::get_commandline_parser(
                command_line_parser(args)
                    .options(desc_cmdline)
                    .positional(pd)
                    .style(unix_style)
                    .extra_parser(detail::option_parser(rtcfg,
                        contains_error_mode(error_mode,
                            commandline_error_mode::ignore_aliases))),
                error_mode & ~commandline_error_mode::ignore_aliases)
                                    .run());

            // collect unregistered options, if needed
            if (unregistered_options) {
                using pika::program_options::collect_unrecognized;
                using pika::program_options::exclude_positional;
                *unregistered_options =
                    collect_unrecognized(opts.options, exclude_positional);
            }

            store(opts, vm);
        }
        else
        {
            // parse command line, allow for unregistered options this point
            parsed_options opts(detail::get_commandline_parser(
                command_line_parser(args)
                    .options(desc_cmdline)
                    .style(unix_style)
                    .extra_parser(detail::option_parser(rtcfg,
                        contains_error_mode(error_mode,
                            commandline_error_mode::ignore_aliases))),
                error_mode & ~commandline_error_mode::ignore_aliases)
                                    .run());

            // collect unregistered options, if needed
            if (unregistered_options) {
                using pika::program_options::collect_unrecognized;
                using pika::program_options::include_positional;
                *unregistered_options =
                    collect_unrecognized(opts.options, include_positional);
            }

            store(opts, vm);
        }

        if (vm.count("pika:help"))
        {
            // collect help information
            if (visible) {
                (*visible)
                    .add(app_options).add(cmdline_options)
                    .add(pika_options)
                    .add(debugging_options).add(config_options)
                ;
            }
            return;
        }

        // clang-format on
        notify(vm);

        detail::handle_generic_config_options(arg0, vm, desc_cfgfile, rtcfg, error_mode);
        detail::handle_config_options(vm, desc_cfgfile, rtcfg, error_mode);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        std::string extract_arg0(std::string const& cmdline)
        {
            std::string::size_type p = cmdline.find_first_of(" \t");
            if (p != std::string::npos) { return cmdline.substr(0, p); }
            return cmdline;
        }
    }    // namespace detail

    void parse_commandline(pika::detail::section const& rtcfg,
        pika::program_options::options_description const& app_options, std::string const& cmdline,
        pika::program_options::variables_map& vm, commandline_error_mode error_mode,
        pika::program_options::options_description* visible,
        std::vector<std::string>* unregistered_options)
    {
        using namespace pika::program_options;
#if defined(PIKA_WINDOWS)
        std::vector<std::string> args = split_winmain(cmdline);
#else
        std::vector<std::string> args = split_unix(cmdline);
#endif
        parse_commandline(rtcfg, app_options, detail::extract_arg0(cmdline), args, vm, error_mode,
            visible, unregistered_options);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string embed_in_quotes(std::string const& s)
    {
        char quote = (s.find_first_of('"') != std::string::npos) ? '\'' : '"';

        if (s.find_first_of("\t ") != std::string::npos) return quote + s + quote;
        return s;
    }

    void add_as_option(std::string& command_line, std::string const& k, std::string const& v)
    {
        command_line += "--" + k;
        if (!v.empty()) command_line += "=" + v;
    }

    std::string reconstruct_command_line(pika::program_options::variables_map const& vm)
    {
        std::string command_line;
        for (auto const& v : vm)
        {
            std::any const& value = v.second.value();
            if (std::any_cast<std::string>(&value))
            {
                add_as_option(command_line, v.first, embed_in_quotes(v.second.as<std::string>()));
                if (!command_line.empty()) command_line += " ";
            }
            else if (std::any_cast<double>(&value))
            {
                add_as_option(command_line, v.first, std::to_string(v.second.as<double>()));
                if (!command_line.empty()) command_line += " ";
            }
            else if (std::any_cast<int>(&value))
            {
                add_as_option(command_line, v.first, std::to_string(v.second.as<int>()));
                if (!command_line.empty()) command_line += " ";
            }
            else if (std::any_cast<std::vector<std::string>>(&value))
            {
                auto const& vec = v.second.as<std::vector<std::string>>();
                for (std::string const& e : vec)
                {
                    add_as_option(command_line, v.first, embed_in_quotes(e));
                    if (!command_line.empty()) command_line += " ";
                }
            }
        }
        return command_line;
    }
}    // namespace pika::detail
