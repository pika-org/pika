//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/command_line_handling/command_line_handling.hpp>
#include <pika/command_line_handling/parse_command_line.hpp>
#include <pika/debugging/attach_debugger.hpp>
#include <pika/functional/detail/reset_function.hpp>
#include <pika/logging.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/string_util/from_string.hpp>
#include <pika/topology/cpu_mask.hpp>
#include <pika/topology/topology.hpp>
#include <pika/type_support/unused.hpp>
#include <pika/util/get_entry_as.hpp>
#include <pika/util/manage_config.hpp>
#include <pika/version.hpp>

#include <boost/tokenizer.hpp>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <spdlog/common.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pika::detail {
    std::string runtime_configuration_string(command_line_handling const& cfg)
    {
        std::ostringstream strm;

        // default scheduler used for this run
        strm << "  {scheduler}: " << cfg.queuing_ << "\n";

        // amount of threads and cores configured for this run
        strm << "  {os-threads}: " << cfg.num_threads_ << "\n";
        strm << "  {cores}: " << cfg.num_cores_ << "\n";

        return strm.str();
    }

    ///////////////////////////////////////////////////////////////////////
    int print_version(std::ostream& out)
    {
        out << std::endl << pika::copyright() << std::endl;
        out << pika::complete_version() << std::endl;
        return 1;
    }

    int print_info(std::ostream& out, command_line_handling const& cfg)
    {
        out << "Static configuration:\n---------------------\n";
        out << pika::configuration_string() << std::endl;

        out << "Runtime configuration:\n----------------------\n";
        out << runtime_configuration_string(cfg) << std::endl;

        return 1;
    }

    ///////////////////////////////////////////////////////////////////////
    inline void encode(std::string& str, char s, char const* r, std::size_t inc = 1ull)
    {
        std::string::size_type pos = 0;
        while ((pos = str.find_first_of(s, pos)) != std::string::npos)
        {
            str.replace(pos, 1, r);
            pos += inc;
        }
    }

    inline std::string encode_string(std::string str)
    {
        encode(str, '\n', "\\n");
        return str;
    }

    inline std::string encode_and_enquote(std::string str)
    {
        encode(str, '\"', "\\\"", 2);
        return enquote(PIKA_MOVE(str));
    }

    ///////////////////////////////////////////////////////////////////////
    std::string handle_process_mask(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, bool use_process_mask)
    {
        std::string mask_string = cfgmap.get_value<std::string>("pika.process_mask", "");

        char const* mask_env = std::getenv("PIKA_PROCESS_MASK");
        if (nullptr != mask_env) { mask_string = mask_env; }

        if (vm.count("pika:process-mask"))
        {
            mask_string = vm["pika:process-mask"].as<std::string>();
        }

#if defined(__APPLE__)
        PIKA_UNUSED(use_process_mask);

        if (!mask_string.empty())
        {
            PIKA_LOG(warn,
                "Explicit process mask is set with --pika:process-mask or PIKA_PROCESS_MASK, but "
                "thread binding is not supported on macOS. The process mask will be ignored.");
            mask_string = "";
        }
#else
        if (!mask_string.empty() && !use_process_mask)
        {
            PIKA_LOG(warn,
                "Explicit process mask is set with --pika:process-mask or PIKA_PROCESS_MASK, but "
                "--pika:ignore-process-mask is also set. The process mask will be ignored.");
        }
#endif

        return mask_string;
    }

    std::string handle_queuing(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::string const& default_)
    {
        // command line options is used preferred
        if (vm.count("pika:queuing")) return vm["pika:queuing"].as<std::string>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::string>("pika.scheduler", default_);
    }

    std::string handle_affinity(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::string const& default_)
    {
        // command line options is used preferred
        if (vm.count("pika:affinity")) return vm["pika:affinity"].as<std::string>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::string>("pika.affinity", default_);
    }

    std::string handle_affinity_bind(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::string const& default_)
    {
        // command line options is used preferred
        if (vm.count("pika:bind"))
        {
            std::string affinity_desc;

            std::vector<std::string> bind_affinity = vm["pika:bind"].as<std::vector<std::string>>();
            for (std::string const& s : bind_affinity)
            {
                if (!affinity_desc.empty()) affinity_desc += ";";
                affinity_desc += s;
            }

            return affinity_desc;
        }

        // use either cfgmap value or default
        return cfgmap.get_value<std::string>("pika.bind", default_);
    }

    std::size_t handle_pu_step(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::size_t default_)
    {
        // command line options is used preferred
        if (vm.count("pika:pu-step")) return vm["pika:pu-step"].as<std::size_t>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::size_t>("pika.pu_step", default_);
    }

    std::size_t handle_pu_offset(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::size_t default_)
    {
        // command line options is used preferred
        if (vm.count("pika:pu-offset")) return vm["pika:pu-offset"].as<std::size_t>();

        // use either cfgmap value or default
        return cfgmap.get_value<std::size_t>("pika.pu_offset", default_);
    }

    std::size_t handle_numa_sensitive(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::size_t default_)
    {
        if (vm.count("pika:numa-sensitive") != 0)
        {
            std::size_t numa_sensitive = vm["pika:numa-sensitive"].as<std::size_t>();
            if (numa_sensitive > 2)
            {
                throw pika::detail::command_line_error(
                    "Invalid argument value for --pika:numa-sensitive. Allowed values are 0, 1, or "
                    "2");
            }
            return numa_sensitive;
        }

        // use either cfgmap value or default
        return cfgmap.get_value<std::size_t>("pika.numa_sensitive", default_);
    }

    ///////////////////////////////////////////////////////////////////////
    std::size_t get_number_of_default_threads(bool use_process_mask)
    {
        if (use_process_mask)
        {
            threads::detail::topology& top = threads::detail::get_topology();
            return threads::detail::count(top.get_cpubind_mask_main_thread());
        }
        else { return threads::detail::hardware_concurrency(); }
    }

    std::size_t get_number_of_default_cores(bool use_process_mask)
    {
        threads::detail::topology& top = threads::detail::get_topology();

        std::size_t num_cores = top.get_number_of_cores();

        if (use_process_mask)
        {
            threads::detail::mask_type proc_mask = top.get_cpubind_mask_main_thread();
            std::size_t num_cores_proc_mask = 0;

            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                threads::detail::mask_type core_mask =
                    top.init_core_affinity_mask_from_core(num_core);
                if (threads::detail::bit_and(core_mask, proc_mask)) { ++num_cores_proc_mask; }
            }

            return num_cores_proc_mask;
        }

        return num_cores;
    }

    ///////////////////////////////////////////////////////////////////////
    std::size_t handle_num_threads(detail::manage_config& cfgmap,
        pika::util::runtime_configuration const& rtcfg, pika::program_options::variables_map& vm,
        bool use_process_mask)
    {
        // If using the process mask we override "cores" and "all" options but
        // keep explicit numeric values.
        std::size_t const init_threads = get_number_of_default_threads(use_process_mask);
        std::size_t const init_cores = get_number_of_default_cores(use_process_mask);

        std::size_t default_threads = init_threads;

        std::string threads_str = cfgmap.get_value<std::string>(
            "pika.os_threads", rtcfg.get_entry("pika.os_threads", std::to_string(default_threads)));

        if ("cores" == threads_str) { default_threads = init_cores; }
        else if ("all" == threads_str) { default_threads = init_threads; }
        else { default_threads = pika::detail::from_string<std::size_t>(threads_str); }

        std::size_t threads = cfgmap.get_value<std::size_t>("pika.os_threads", default_threads);

        if (vm.count("pika:threads"))
        {
            threads_str = vm["pika:threads"].as<std::string>();
            if ("all" == threads_str) { threads = init_threads; }
            else if ("cores" == threads_str) { threads = init_cores; }
            else { threads = pika::detail::from_string<std::size_t>(threads_str); }

            if (threads == 0)
            {
                throw pika::detail::command_line_error(
                    "Number of --pika:threads must be greater than 0");
            }

#if defined(PIKA_HAVE_MAX_CPU_COUNT)
            if (threads > PIKA_HAVE_MAX_CPU_COUNT)
            {
                // clang-format off
                    throw pika::detail::command_line_error("Requested more than "
                        PIKA_PP_STRINGIZE(PIKA_HAVE_MAX_CPU_COUNT)" --pika:threads "
                        "to use for this application, use the option "
                        "-DPIKA_WITH_MAX_CPU_COUNT=<N> when configuring pika.");
                // clang-format on
            }
#endif
        }

        // make sure minimal requested number of threads is observed
        std::size_t min_os_threads =
            cfgmap.get_value<std::size_t>("pika.force_min_os_threads", threads);

        if (min_os_threads == 0)
        {
            throw pika::detail::command_line_error(
                "Number of pika.force_min_os_threads must be greater than 0");
        }

#if defined(PIKA_HAVE_MAX_CPU_COUNT)
        if (min_os_threads > PIKA_HAVE_MAX_CPU_COUNT)
        {
            throw pika::detail::command_line_error("Requested more than " PIKA_PP_STRINGIZE(
                PIKA_HAVE_MAX_CPU_COUNT) " pika.force_min_os_threads to use for this application, "
                                         "use the option -DPIKA_WITH_MAX_CPU_COUNT=<N> when "
                                         "configuring pika.");
        }
#endif

        threads = (std::max)(threads, min_os_threads);

        return threads;
    }

    std::size_t handle_num_cores(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::size_t num_threads, bool use_process_mask)
    {
        std::string cores_str = cfgmap.get_value<std::string>("pika.cores", "");
        if ("all" == cores_str)
        {
            cfgmap.config_["pika.cores"] =
                std::to_string(get_number_of_default_cores(use_process_mask));
        }

        std::size_t num_cores = cfgmap.get_value<std::size_t>("pika.cores", num_threads);
        if (vm.count("pika:cores"))
        {
            cores_str = vm["pika:cores"].as<std::string>();
            if ("all" == cores_str) { num_cores = get_number_of_default_cores(use_process_mask); }
            else { num_cores = pika::detail::from_string<std::size_t>(cores_str); }
        }

        return num_cores;
    }

    ///////////////////////////////////////////////////////////////////////
    void command_line_handling::check_affinity_domain() const
    {
        if (affinity_domain_ != "pu")
        {
            if (0 != std::string("pu").find(affinity_domain_) &&
                0 != std::string("core").find(affinity_domain_) &&
                0 != std::string("socket").find(affinity_domain_) &&
                0 != std::string("machine").find(affinity_domain_))
            {
                throw pika::detail::command_line_error(
                    "Invalid command line option --pika:affinity, value must be one of: pu, core, "
                    "socket, or machine.");
            }
        }
    }

    void command_line_handling::check_affinity_description() const
    {
        if (affinity_bind_.empty()) { return; }

        if (!(pu_offset_ == std::size_t(-1) || pu_offset_ == std::size_t(0)) || pu_step_ != 1 ||
            affinity_domain_ != "pu")
        {
            throw pika::detail::command_line_error(
                "Command line option --pika:bind should not be used with --pika:pu-step, "
                "--pika:pu-offset, or --pika:affinity.");
        }
    }

    void command_line_handling::check_pu_offset() const
    {
        if (pu_offset_ != std::size_t(-1) &&
            pu_offset_ >= pika::threads::detail::hardware_concurrency())
        {
            throw pika::detail::command_line_error(
                "Invalid command line option --pika:pu-offset, value must be smaller than number "
                "of available processing units.");
        }
    }

    void command_line_handling::check_pu_step() const
    {
        if (pika::threads::detail::hardware_concurrency() > 1 &&
            (pu_step_ == 0 || pu_step_ >= pika::threads::detail::hardware_concurrency()))
        {
            throw pika::detail::command_line_error(
                "Invalid command line option --pika:pu-step, value must be non-zero and smaller "
                "than number of available processing units.");
        }
    }

#if defined(PIKA_HAVE_MPI)
    bool handle_enable_mpi_pool(detail::manage_config& cfgmap,
        pika::util::runtime_configuration const& rtcfg, pika::program_options::variables_map& vm)
    {
        bool enable_mpi_pool = cfgmap.get_value<bool>("pika.mpi.enable_pool",
            pika::detail::get_entry_as<bool>(rtcfg, "pika.mpi.enable_pool", false));

        enable_mpi_pool |= vm.count("pika:mpi-enable-pool");
        return enable_mpi_pool;
    }

    std::size_t handle_mpi_completion_mode(detail::manage_config& cfgmap,
        pika::util::runtime_configuration const& rtcfg, pika::program_options::variables_map& vm)
    {
        std::size_t completion_mode = cfgmap.get_value<std::size_t>("pika.mpi.completion_mode",
            pika::detail::get_entry_as<std::size_t>(rtcfg, "pika.mpi.completion_mode", 0));

        if (vm.count("pika:mpi-completion-mode"))
        {
            completion_mode = vm["pika:mpi-completion-mode"].as<std::size_t>();
        }

        return completion_mode;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::handle_arguments(detail::manage_config& cfgmap,
        pika::program_options::variables_map& vm, std::vector<std::string>& ini_config)
    {
        bool debug_clp = vm.count("pika:debug-clp");

        if (vm.count("pika:ini"))
        {
            std::vector<std::string> cfg = vm["pika:ini"].as<std::vector<std::string>>();
            std::copy(cfg.begin(), cfg.end(), std::back_inserter(ini_config));
            cfgmap.add(cfg);
        }

        use_process_mask_ =
#if defined(__APPLE__)
            false;
#else
            !((cfgmap.get_value<int>("pika.ignore_process_mask", 0) > 0) ||
                (vm.count("pika:ignore-process-mask") > 0));
#endif

        ini_config.emplace_back("pika.ignore_process_mask!=" + std::to_string(!use_process_mask_));

        process_mask_ = handle_process_mask(cfgmap, vm, use_process_mask_);
        ini_config.emplace_back("pika.process_mask!=" + process_mask_);
        if (!process_mask_.empty())
        {
            auto const m = from_string<threads::detail::mask_type>(process_mask_);
            threads::detail::get_topology().set_cpubind_mask_main_thread(m);
        }

        // handle setting related to schedulers
        queuing_ = detail::handle_queuing(cfgmap, vm, "local-priority-fifo");
        ini_config.emplace_back("pika.scheduler=" + queuing_);

        affinity_domain_ = detail::handle_affinity(cfgmap, vm, "pu");
        ini_config.emplace_back("pika.affinity=" + affinity_domain_);

        check_affinity_domain();

        affinity_bind_ = detail::handle_affinity_bind(cfgmap, vm, "");
        if (!affinity_bind_.empty())
        {
#if defined(__APPLE__)
            if (affinity_bind_ != "none")
            {
                PIKA_LOG(warn,
                    "Thread binding set to \"{}\" but thread binding is not supported on macOS. "
                    "Ignoring option.",
                    affinity_bind_);
            }
            affinity_bind_ = "";
#else
            ini_config.emplace_back("pika.bind!=" + affinity_bind_);
#endif
        }

        pu_step_ = detail::handle_pu_step(cfgmap, vm, 1);
#if defined(__APPLE__)
        if (pu_step_ != 1)
        {
            PIKA_LOG(warn,
                "PU step set to \"{}\" but thread binding is not supported on macOS. "
                "Ignoring option.",
                pu_step_);
            pu_step_ = 1;
        }
#endif
        ini_config.emplace_back("pika.pu_step=" + std::to_string(pu_step_));

        check_pu_step();

        pu_offset_ = detail::handle_pu_offset(cfgmap, vm, std::size_t(-1));

        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (pu_offset_ != std::size_t(-1))
        {
#if defined(__APPLE__)
            PIKA_LOG(warn,
                "PU offset set to \"{}\" but thread binding is not supported on macOS. Ignoring "
                "option.",
                pu_offset_);
            pu_offset_ = std::size_t(-1);
            ini_config.emplace_back("pika.pu_offset=0");
#else
            ini_config.emplace_back("pika.pu_offset=" + std::to_string(pu_offset_));
#endif
        }
        else { ini_config.emplace_back("pika.pu_offset=0"); }

        check_pu_offset();

        numa_sensitive_ = detail::handle_numa_sensitive(cfgmap, vm, affinity_bind_.empty() ? 0 : 1);
        ini_config.emplace_back("pika.numa_sensitive=" + std::to_string(numa_sensitive_));

        // default affinity mode is now 'balanced' (only if no pu-step or
        // pu-offset is given)
        if (pu_step_ == 1 && pu_offset_ == std::size_t(-1) && affinity_bind_.empty())
        {
#if defined(__APPLE__)
            affinity_bind_ = "none";
#else
            affinity_bind_ = "balanced";
#endif
            ini_config.emplace_back("pika.bind!=" + affinity_bind_);
        }

        check_affinity_description();

        // handle number of cores and threads
        num_threads_ = detail::handle_num_threads(cfgmap, rtcfg_, vm, use_process_mask_);
        num_cores_ = detail::handle_num_cores(cfgmap, vm, num_threads_, use_process_mask_);

        // Set number of cores and OS threads in configuration.
        ini_config.emplace_back("pika.os_threads=" + std::to_string(num_threads_));
        ini_config.emplace_back("pika.cores=" + std::to_string(num_cores_));

        if (vm_.count("pika:high-priority-threads"))
        {
            std::size_t num_high_priority_queues =
                vm_["pika:high-priority-threads"].as<std::size_t>();
            if (num_high_priority_queues != std::size_t(-1) &&
                num_high_priority_queues > num_threads_)
            {
                throw pika::detail::command_line_error(
                    "Invalid command line option: number of high priority threads "
                    "(--pika:high-priority-threads), should not be larger than number of threads "
                    "(--pika:threads)");
            }

            if (!(queuing_ == "local-priority" || queuing_ == "abp-priority"))
            {
                throw pika::detail::command_line_error(
                    "Invalid command line option --pika:high-priority-threads, valid for "
                    "--pika:queuing=local-priority and --pika:queuing=abp-priority only");
            }

            ini_config.emplace_back("pika.thread_queue.high_priority_queues!=" +
                std::to_string(num_high_priority_queues));
        }

#if defined(PIKA_HAVE_MPI)
        bool enable_mpi_pool = detail::handle_enable_mpi_pool(cfgmap, rtcfg_, vm);
        ini_config.emplace_back("pika.mpi.enable_pool=" + std::to_string(enable_mpi_pool));

        std::size_t const mpi_completion_mode =
            detail::handle_mpi_completion_mode(cfgmap, rtcfg_, vm);
        ini_config.emplace_back("pika.mpi.completion_mode=" + std::to_string(mpi_completion_mode));
#endif

        update_logging_settings(vm, ini_config);

        if (debug_clp)
        {
            std::cerr << "Configuration before runtime start:\n";
            std::cerr << "-----------------------------------\n";
            for (std::string const& s : ini_config) { std::cerr << s << std::endl; }
            std::cerr << "-----------------------------------\n";
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::update_logging_settings(
        pika::program_options::variables_map& vm, std::vector<std::string>& ini_config)
    {
        if (vm.count("pika:log-destination"))
        {
            ini_config.emplace_back(
                "pika.log.destination=" + vm["pika:log-destination"].as<std::string>());
        }

        if (vm.count("pika:log-level"))
        {
            ini_config.emplace_back("pika.log.level=" +
                std::to_string(
                    vm["pika:log-level"].as<std::underlying_type_t<spdlog::level::level_enum>>()));
        }

        if (vm.count("pika:log-format"))
        {
            ini_config.emplace_back("pika.log.format=" + vm["pika:log-format"].as<std::string>());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::store_command_line(int argc, char const* const* argv)
    {
        // Collect the command line for diagnostic purposes.
        std::string command;
        std::string cmd_line;
        std::string options;
        for (int i = 0; i < argc; ++i)
        {
            // quote only if it contains whitespace
            std::string arg = detail::encode_and_enquote(argv[i]);    //-V108

            cmd_line += arg;
            if (i == 0) { command = arg; }
            else { options += " " + arg; }

            if ((i + 1) != argc) { cmd_line += " "; }
        }

        // Store the program name and the command line.
        ini_config_.emplace_back("pika.cmd_line!=" + cmd_line);
        ini_config_.emplace_back("pika.commandline.command!=" + command);
        ini_config_.emplace_back("pika.commandline.options!=" + options);
    }

    ///////////////////////////////////////////////////////////////////////////
    void command_line_handling::store_unregistered_options(
        std::string const& cmd_name, std::vector<std::string> const& unregistered_options)
    {
        std::string unregistered_options_cmd_line;

        if (!unregistered_options.empty())
        {
            using iterator_type = std::vector<std::string>::const_iterator;

            iterator_type end = unregistered_options.end();
            // Silence bogus warning from GCC 12:
            // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 120000
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wrestrict"
#endif
            for (iterator_type it = unregistered_options.begin(); it != end; ++it)
                unregistered_options_cmd_line += " " + detail::encode_and_enquote(*it);
#if defined(PIKA_GCC_VERSION) && PIKA_GCC_VERSION >= 120000
# pragma GCC diagnostic pop
#endif

            ini_config_.emplace_back("pika.unknown_cmd_line!=" +
                detail::encode_and_enquote(cmd_name) + unregistered_options_cmd_line);
        }

        ini_config_.emplace_back("pika.program_name!=" + cmd_name);
        ini_config_.emplace_back("pika.reconstructed_cmd_line!=" + encode_and_enquote(cmd_name) +
            " " + reconstruct_command_line(vm_) + " " + unregistered_options_cmd_line);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool command_line_handling::handle_help_options(
        pika::program_options::options_description const& help)
    {
        if (vm_.count("pika:help"))
        {
            std::cout << help << std::endl;
            return true;
        }
        return false;
    }

    void command_line_handling::handle_attach_debugger()
    {
#if defined(_POSIX_VERSION) || defined(PIKA_WINDOWS)
        if (vm_.count("pika:attach-debugger"))
        {
            std::string option = vm_["pika:attach-debugger"].as<std::string>();
            if (option != "off" && option != "startup" && option != "exception" &&
                option != "test-failure")
            {
                PIKA_LOG(warn,
                    "pika::init: command line warning: --pika:attach-debugger: invalid option: "
                    "\"{}\". Allowed values are \"off\", \"startup\", \"exception\" or "
                    "\"test-failure\"",
                    option);
            }
            else
            {
                if (option == "startup") { debug::detail::attach_debugger(); }
                else if (option == "exception")
                {
                    // Signal handlers need to be installed to be able to attach
                    // the debugger on uncaught exceptions
                    ini_config_.emplace_back("pika.install_signal_handlers!=1");
                }

                ini_config_.emplace_back("pika.attach_debugger!=" + option);
            }
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // separate command line arguments from configuration settings
    std::vector<std::string> command_line_handling::preprocess_config_settings(
        int argc, char const* const* argv)
    {
        std::vector<std::string> options;
        options.reserve(static_cast<std::size_t>(argc) + ini_config_.size());

        // extract all command line arguments from configuration settings and
        // remove them from this list
        auto it = std::stable_partition(ini_config_.begin(), ini_config_.end(),
            [](std::string const& e) { return e.find("--pika:") != 0; });

        std::move(it, ini_config_.end(), std::back_inserter(options));
        ini_config_.erase(it, ini_config_.end());

        // store the command line options that came from the configuration
        // settings in the registry
        if (!options.empty())
        {
            std::string config_options;
            for (auto const& option : options) { config_options += " " + option; }

            rtcfg_.add_entry("pika.commandline.config_options", config_options);
        }

        // now append all original command line options
        for (int i = 1; i != argc; ++i) { options.emplace_back(argv[i]); }

        return options;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<std::string> prepend_options(std::vector<std::string>&& args, std::string&& options)
    {
        if (options.empty()) { return PIKA_MOVE(args); }

        using tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;
        boost::escaped_list_separator<char> sep('\\', ' ', '\"');
        tokenizer tok(options, sep);

        std::vector<std::string> result(tok.begin(), tok.end());
        std::move(args.begin(), args.end(), std::back_inserter(result));
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    command_line_handling_result command_line_handling::call(
        pika::program_options::options_description const& desc_cmdline, int argc,
        char const* const* argv)
    {
        // set the flag signaling that command line parsing has been done
        cmd_line_parsed_ = true;

        // separate command line arguments from configuration settings
        std::vector<std::string> args = preprocess_config_settings(argc, argv);

        detail::manage_config cfgmap(ini_config_);

        // insert the pre-configured ini settings before loading modules
        for (std::string const& e : ini_config_)
            rtcfg_.parse("<user supplied config>", e, true, false);

        // support re-throwing command line exceptions for testing purposes
        commandline_error_mode error_mode = commandline_error_mode::allow_unregistered;
        if (cfgmap.get_value("pika.commandline.rethrow_errors", 0) != 0)
        {
            error_mode |= commandline_error_mode::rethrow_on_error;
        }

        // The cfg registry may hold command line options to prepend to the
        // real command line.
        std::string prepend_command_line = rtcfg_.get_entry("pika.commandline.prepend_options");

        args = prepend_options(PIKA_MOVE(args), PIKA_MOVE(prepend_command_line));

        // Initial analysis of the command line options. This is
        // preliminary as it will not take into account any aliases as
        // defined in any of the runtime configuration files.
        {
            // Boost V1.47 and before do not properly reset a variables_map
            // when calling vm.clear(). We work around that problems by
            // creating a separate instance just for the preliminary
            // command line handling.
            pika::program_options::variables_map prevm;
            parse_commandline(rtcfg_, desc_cmdline, argv[0], args, prevm, error_mode);

            // handle all --pika:foo options
            std::vector<std::string> ini_config;    // discard
            handle_arguments(cfgmap, prevm, ini_config);

            // re-initialize runtime configuration object
            if (prevm.count("pika:config"))
                rtcfg_.reconfigure(prevm["pika:config"].as<std::string>());
            else
                rtcfg_.reconfigure("");

            // Make sure any aliases defined on the command line get used
            // for the option analysis below.
            std::vector<std::string> cfg;
            if (prevm.count("pika:ini"))
            {
                cfg = prevm["pika:ini"].as<std::vector<std::string>>();
                cfgmap.add(cfg);
            }

            // append ini options from command line
            std::copy(ini_config_.begin(), ini_config_.end(), std::back_inserter(cfg));

            // set logging options from command line options
            std::vector<std::string> ini_config_logging;
            update_logging_settings(prevm, ini_config_logging);

            std::copy(
                ini_config_logging.begin(), ini_config_logging.end(), std::back_inserter(cfg));

            rtcfg_.reconfigure(cfg);
        }

        // Re-run program option analysis, ini settings (such as aliases)
        // will be considered now.

        // Now re-parse the command line using the node number (if given).
        // This will additionally detect any --pika:N:foo options.
        pika::program_options::options_description help;
        std::vector<std::string> unregistered_options;

        parse_commandline(rtcfg_, desc_cmdline, argv[0], args, vm_,
            error_mode | commandline_error_mode::report_missing_config_file, &help,
            &unregistered_options);

        // break into debugger, if requested
        handle_attach_debugger();

        // handle all --pika:foo and --pika:*:foo options
        handle_arguments(cfgmap, vm_, ini_config_);

        // store unregistered command line and arguments
        store_command_line(argc, argv);
        store_unregistered_options(argv[0], unregistered_options);

        // add all remaining ini settings to the global configuration
        rtcfg_.reconfigure(ini_config_);

        // help can be printed only after the runtime mode has been set
        if (handle_help_options(help)) { return command_line_handling_result::exit; }

        // print version/copyright information
        if (vm_.count("pika:version"))
        {
            if (!version_printed_)
            {
                detail::print_version(std::cout);
                version_printed_ = true;
            }

            return command_line_handling_result::exit;
        }

        // print configuration information (static and dynamic)
        if (vm_.count("pika:info"))
        {
            if (!info_printed_)
            {
                detail::print_info(std::cout, *this);
                info_printed_ = true;
            }

            return command_line_handling_result::exit;
        }

        // Print a warning about process masks resulting in only one worker
        // thread, but only do so if that would not be the default on the given
        // system and it was not given explicitly to --pika:threads.
        if (use_process_mask_)
        {
            bool const command_line_arguments_given =
                vm_.count("pika:threads") != 0 || vm_.count("pika:cores") != 0;
            if (num_threads_ == 1 && get_number_of_default_threads(false) != 1 &&
                !command_line_arguments_given)
            {
                PIKA_LOG(warn,
                    "The pika runtime will be started with only one worker thread because the "
                    "process mask has restricted the available resources to only one thread. If "
                    "this is unintentional make sure the process mask contains the resources "
                    "you need or use --pika:ignore-process-mask to use all resources. Use "
                    "--pika:print-bind to print the thread bindings used by pika.");
            }
            else if (num_cores_ == 1 && get_number_of_default_cores(false) != 1 &&
                !command_line_arguments_given)
            {
                PIKA_LOG(warn,
                    "The pika runtime will be started on only one core with {} worker threads "
                    "because the process mask has restricted the available resources to only one "
                    "core. If this is unintentional make sure the process mask contains the "
                    "resources you need or use --pika:ignore-process-mask to use all resources. "
                    "Use --pika:print-bind to print the thread bindings used by pika.",
                    num_threads_);
            }
        }

        return command_line_handling_result::success;
    }
}    // namespace pika::detail
