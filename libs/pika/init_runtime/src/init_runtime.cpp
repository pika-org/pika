//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/command_line_handling/command_line_handling.hpp>
#include <pika/coroutines/detail/context_impl.hpp>
#include <pika/detail/filesystem.hpp>
#include <pika/execution_base/detail/spinlock_deadlock_detection.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/function.hpp>
#include <pika/init_runtime/detail/init_logging.hpp>
#include <pika/init_runtime/init_runtime.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/logging.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/timing.hpp>
#include <pika/program_options/parsers.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/resource_partitioner/partitioner.hpp>
#include <pika/runtime/config_entry.hpp>
#include <pika/runtime/custom_exception_info.hpp>
#include <pika/runtime/debugging.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/runtime_handlers.hpp>
#include <pika/runtime/shutdown_function.hpp>
#include <pika/runtime/startup_function.hpp>
#include <pika/string_util/classification.hpp>
#include <pika/string_util/split.hpp>
#include <pika/threading/thread.hpp>
#include <pika/threading_base/detail/get_default_pool.hpp>
#include <pika/type_support/pack.hpp>
#include <pika/type_support/unused.hpp>
#include <pika/util/get_entry_as.hpp>
#include <pika/version.hpp>

#if defined(PIKA_HAVE_HIP)
# include <hip/hip_runtime.h>
# include <whip.hpp>
#endif

#if defined(__bgq__)
# include <cstdlib>
#endif

#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if !defined(PIKA_WINDOWS)
# include <signal.h>
#endif

namespace pika {
    namespace detail {

        int init_helper(pika::program_options::variables_map& /*vm*/,
            pika::util::detail::function<int(int, char**)> const& f)
        {
            std::string cmdline(pika::detail::get_config_entry("pika.reconstructed_cmd_line", ""));

            using namespace pika::program_options;
#if defined(PIKA_WINDOWS)
            std::vector<std::string> args = split_winmain(cmdline);
#else
            std::vector<std::string> args = split_unix(cmdline);
#endif

            // Copy all arguments which are not pika related to a temporary array
            std::vector<char*> argv(args.size() + 1);
            std::size_t argcount = 0;
            for (std::size_t i = 0; i != args.size(); ++i)
            {
                if (0 != args[i].find("--pika:"))
                {
                    argv[argcount++] = const_cast<char*>(args[i].data());
                }
                else if (7 == args[i].find("positional", 7))
                {
                    std::string::size_type p = args[i].find_first_of('=');
                    if (p != std::string::npos)
                    {
                        args[i] = args[i].substr(p + 1);
                        argv[argcount++] = const_cast<char*>(args[i].data());
                    }
                }
            }

            // add a single nullptr in the end as some application rely on that
            argv[argcount] = nullptr;

            // Invoke custom startup functions
            return f(static_cast<int>(argcount), argv.data());
        }

        void activate_global_options(detail::command_line_handling& cmdline)
        {
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
            pika::threads::coroutines::detail::posix::use_guard_pages =
                cmdline.rtcfg_.use_stack_guard_pages();
#endif
#ifdef PIKA_HAVE_VERIFY_LOCKS
            if (cmdline.rtcfg_.enable_lock_detection())
            {
                pika::util::enable_lock_detection();
                pika::util::trace_depth_lock_detection(cmdline.rtcfg_.trace_depth());
            }
            else { pika::util::disable_lock_detection(); }
#endif
#ifdef PIKA_HAVE_THREAD_DEADLOCK_DETECTION
            threads::detail::set_deadlock_detection_enabled(
                cmdline.rtcfg_.enable_deadlock_detection());
#endif
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
            pika::util::detail::set_spinlock_break_on_deadlock_enabled(
                cmdline.rtcfg_.enable_spinlock_deadlock_detection());
            pika::util::detail::set_spinlock_deadlock_detection_limit(
                cmdline.rtcfg_.get_spinlock_deadlock_detection_limit());
            pika::util::detail::set_spinlock_deadlock_warning_limit(
                cmdline.rtcfg_.get_spinlock_deadlock_warning_limit());
#endif
            init_logging(cmdline.rtcfg_);
        }

        struct dump_config
        {
            explicit dump_config(pika::detail::runtime const& rt)
              : rt_(std::cref(rt))
            {
            }

            void operator()() const
            {
                std::cout << "Configuration after runtime start:\n";
                std::cout << "----------------------------------\n";
                rt_.get().get_config().dump(0, std::cout);
                std::cout << "----------------------------------\n";
            }

            std::reference_wrapper<pika::detail::runtime const> rt_;
        };

        ///////////////////////////////////////////////////////////////////////
        void add_startup_functions(pika::detail::runtime& rt,
            pika::program_options::variables_map& vm, startup_function_type startup,
            shutdown_function_type shutdown)
        {
            if (vm.count("pika:app-config"))
            {
                std::string config(vm["pika:app-config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }

            if (!!startup) rt.add_startup_function(PIKA_MOVE(startup));

            if (!!shutdown) rt.add_shutdown_function(PIKA_MOVE(shutdown));

            if (vm.count("pika:dump-config-initial"))
            {
                std::cout << "Configuration after runtime construction:\n";
                std::cout << "-----------------------------------------\n";
                rt.get_config().dump(0, std::cout);
                std::cout << "-----------------------------------------\n";
            }

            if (vm.count("pika:dump-config")) rt.add_startup_function(dump_config(rt));
        }

        ///////////////////////////////////////////////////////////////////////
        int run(pika::detail::runtime& rt,
            pika::util::detail::function<int(pika::program_options::variables_map& vm)> const& f,
            pika::program_options::variables_map& vm, startup_function_type startup,
            shutdown_function_type shutdown)
        {
            add_startup_functions(rt, vm, PIKA_MOVE(startup), PIKA_MOVE(shutdown));

            // Run this runtime instance using the given function f.
            if (!f.empty()) return rt.run(pika::util::detail::bind_front(f, vm));

            // Run this runtime instance without a pika_main
            return rt.run();
        }

        int start(pika::detail::runtime& rt,
            pika::util::detail::function<int(pika::program_options::variables_map& vm)> const& f,
            pika::program_options::variables_map& vm, startup_function_type startup,
            shutdown_function_type shutdown)
        {
            add_startup_functions(rt, vm, PIKA_MOVE(startup), PIKA_MOVE(shutdown));

            if (!f.empty())
            {
                // Run this runtime instance using the given function f.
                return rt.start(pika::util::detail::bind_front(f, vm));
            }

            // Run this runtime instance without a pika_main
            return rt.start();
        }

        int run_or_start(bool blocking, std::unique_ptr<pika::detail::runtime> rt,
            detail::command_line_handling& cfg, startup_function_type startup,
            shutdown_function_type shutdown)
        {
            if (blocking)
            {
                return run(*rt, cfg.pika_main_f_, cfg.vm_, PIKA_MOVE(startup), PIKA_MOVE(shutdown));
            }

            // non-blocking version
            start(*rt, cfg.pika_main_f_, cfg.vm_, PIKA_MOVE(startup), PIKA_MOVE(shutdown));

            // pointer to runtime is stored in TLS
            [[maybe_unused]] pika::detail::runtime* p = rt.release();

            return 0;
        }

        ////////////////////////////////////////////////////////////////////////
        void init_environment(detail::command_line_handling& cmdline)
        {
            PIKA_UNUSED(pika::detail::filesystem::initial_path());

            pika::detail::set_assertion_handler(&pika::detail::assertion_handler);
            pika::detail::set_custom_exception_info_handler(&pika::detail::custom_exception_info);
            pika::detail::set_pre_exception_handler(&pika::detail::pre_exception_handler);
            pika::set_thread_termination_handler(
                [](std::exception_ptr const& e) { report_error(e); });
            pika::detail::set_get_full_build_string(&pika::full_build_string);
#if defined(PIKA_HAVE_VERIFY_LOCKS)
            pika::util::set_registered_locks_error_handler(
                &pika::detail::registered_locks_error_handler);
            pika::util::set_register_locks_predicate(&pika::detail::register_locks_predicate);
#endif
            if (pika::detail::get_entry_as<bool>(
                    cmdline.rtcfg_, "pika.install_signal_handlers", false))
            {
                set_signal_handlers();
            }

            pika::threads::detail::set_get_default_pool(&pika::detail::get_default_pool);

#if defined(__bgq__) || defined(__bgqion__)
            unsetenv("LANG");
            unsetenv("LC_CTYPE");
            unsetenv("LC_NUMERIC");
            unsetenv("LC_TIME");
            unsetenv("LC_COLLATE");
            unsetenv("LC_MONETARY");
            unsetenv("LC_MESSAGES");
            unsetenv("LC_PAPER");
            unsetenv("LC_NAME");
            unsetenv("LC_ADDRESS");
            unsetenv("LC_TELEPHONE");
            unsetenv("LC_MEASUREMENT");
            unsetenv("LC_IDENTIFICATION");
            unsetenv("LC_ALL");
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        int run_or_start(
            pika::util::detail::function<int(pika::program_options::variables_map& vm)> const& f,
            int argc, const char* const* argv, init_params const& params, bool blocking)
        {
            if (pika::detail::get_runtime_ptr() != nullptr)
            {
                PIKA_THROW_EXCEPTION(
                    pika::error::invalid_status, "pika::start/init", "runtime already initialized");
            }

            pika::detail::command_line_handling cmdline{
                pika::util::runtime_configuration(argv[0]), params.cfg, f};

            auto cmdline_result = cmdline.call(params.desc_cmdline, argc, argv);

            pika::detail::affinity_data affinity_data{};
            affinity_data.init(
                pika::detail::get_entry_as<std::size_t>(cmdline.rtcfg_, "pika.os_threads", 0),
                pika::detail::get_entry_as<std::size_t>(cmdline.rtcfg_, "pika.cores", 0),
                pika::detail::get_entry_as<std::size_t>(cmdline.rtcfg_, "pika.pu_offset", 0),
                pika::detail::get_entry_as<std::size_t>(cmdline.rtcfg_, "pika.pu_step", 0), 0,
                cmdline.rtcfg_.get_entry("pika.affinity", ""),
                cmdline.rtcfg_.get_entry("pika.bind", ""),
                !pika::detail::get_entry_as<bool>(
                    cmdline.rtcfg_, "pika.ignore_process_mask", false));

            pika::resource::partitioner rp = pika::resource::detail::make_partitioner(
                params.rp_mode, cmdline.rtcfg_, affinity_data);

            activate_global_options(cmdline);

            init_environment(cmdline);

            switch (cmdline_result)
            {
            case command_line_handling_result::success: break;
            case command_line_handling_result::exit: return 0;
            }

            // If thread_pools initialization in user main
            if (params.rp_callback) { params.rp_callback(rp, cmdline.vm_); }

            // Setup all internal parameters of the resource_partitioner
            rp.configure_pools();

#if defined(PIKA_HAVE_HIP)
            PIKA_LOG(info, "run_local: initialize HIP");
            whip::check_error(hipInit(0));
#endif

            // Initialize and start the pika runtime.
            PIKA_LOG(info, "run_local: create runtime");

            // Build and configure this runtime instance.
            std::unique_ptr<pika::detail::runtime> rt;

            // Command line handling should have updated this by now.
            PIKA_LOG(info, "creating local runtime");
            rt.reset(new pika::detail::runtime(cmdline.rtcfg_, true));

            return run_or_start(blocking, PIKA_MOVE(rt), cmdline, PIKA_MOVE(params.startup),
                PIKA_MOVE(params.shutdown));
        }

        int init_start_impl(
            pika::util::detail::function<int(pika::program_options::variables_map&)> f, int argc,
            const char* const* argv, init_params const& params, bool blocking)
        {
            if (argc == 0 || argv == nullptr)
            {
                argc = dummy_argc;
                argv = dummy_argv;
            }

#if defined(__FreeBSD__)
            freebsd_environ = environ;
#endif
            // set a handler for std::abort
            [[maybe_unused]] auto signal_handler = std::signal(SIGABRT, pika::detail::on_abort);
            [[maybe_unused]] auto exit_result = std::atexit(pika::detail::on_exit);
#if defined(PIKA_HAVE_CXX11_STD_QUICK_EXIT)
            [[maybe_unused]] auto quick_exit_result = std::at_quick_exit(pika::detail::on_exit);
#endif
            return run_or_start(f, argc, argv, params, blocking);
        }
    }    // namespace detail

    int init(std::function<int(pika::program_options::variables_map&)> f, int argc,
        const char* const* argv, init_params const& params)
    {
        return detail::init_start_impl(PIKA_MOVE(f), argc, argv, params, true);
    }

    int init(std::function<int(int, char**)> f, int argc, const char* const* argv,
        init_params const& params)
    {
        pika::util::detail::function<int(pika::program_options::variables_map&)> main_f =
            pika::util::detail::bind_back(pika::detail::init_helper, f);
        return detail::init_start_impl(PIKA_MOVE(main_f), argc, argv, params, true);
    }

    int init(std::function<int()> f, int argc, const char* const* argv, init_params const& params)
    {
        pika::util::detail::function<int(pika::program_options::variables_map&)> main_f =
            pika::util::detail::bind(f);
        return detail::init_start_impl(PIKA_MOVE(main_f), argc, argv, params, true);
    }

    int init(std::nullptr_t, int argc, const char* const* argv, init_params const& params)
    {
        pika::util::detail::function<int(pika::program_options::variables_map&)> main_f;
        return detail::init_start_impl(PIKA_MOVE(main_f), argc, argv, params, true);
    }

    void start(std::function<int(pika::program_options::variables_map&)> f, int argc,
        const char* const* argv, init_params const& params)
    {
        if (detail::init_start_impl(PIKA_MOVE(f), argc, argv, params, false) != 0)
        {
            PIKA_UNREACHABLE;
        }
    }

    void start(std::function<int(int, char**)> f, int argc, const char* const* argv,
        init_params const& params)
    {
        pika::util::detail::function<int(pika::program_options::variables_map&)> main_f =
            pika::util::detail::bind_back(pika::detail::init_helper, f);
        if (detail::init_start_impl(PIKA_MOVE(main_f), argc, argv, params, false) != 0)
        {
            PIKA_UNREACHABLE;
        }
    }

    void start(std::function<int()> f, int argc, const char* const* argv, init_params const& params)
    {
        pika::util::detail::function<int(pika::program_options::variables_map&)> main_f =
            pika::util::detail::bind(f);
        if (detail::init_start_impl(PIKA_MOVE(main_f), argc, argv, params, false) != 0)
        {
            PIKA_UNREACHABLE;
        }
    }

    void start(std::nullptr_t, int argc, const char* const* argv, init_params const& params)
    {
        pika::util::detail::function<int(pika::program_options::variables_map&)> main_f;
        if (detail::init_start_impl(PIKA_MOVE(main_f), argc, argv, params, false) != 0)
        {
            PIKA_UNREACHABLE;
        }
    }

    void start(int argc, const char* const* argv, init_params const& params)
    {
        pika::util::detail::function<int(pika::program_options::variables_map&)> main_f;
        if (detail::init_start_impl(PIKA_MOVE(main_f), argc, argv, params, false) != 0)
        {
            PIKA_UNREACHABLE;
        }
    }

    void finalize()
    {
        if (!pika::detail::is_running())
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::finalize",
                "the runtime system is not active (did you already call finalize?)");
        }

        pika::detail::runtime* rt = pika::detail::get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::finalize",
                "the runtime system is not active (did you already call pika::stop?)");
        }

        rt->finalize();
    }

    int stop()
    {
        if (threads::detail::get_self_ptr())
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::stop",
                "this function cannot be called from a pika thread");
        }
        // take ownership!
        std::unique_ptr<pika::detail::runtime> rt(pika::detail::get_runtime_ptr());
        if (nullptr == rt.get())
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::stop",
                "the runtime system is not active (did you already call pika::stop?)");
        }

        int result = rt->wait();

        rt->stop();
        rt->rethrow_exception();

        return result;
    }

    void wait()
    {
        pika::detail::runtime* rt = pika::detail::get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::wait",
                "the runtime system is not active (did you already call pika::stop?)");
        }

        rt->get_thread_manager().wait();
    }

    void suspend()
    {
        if (threads::detail::get_self_ptr())
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::suspend",
                "this function cannot be called from a pika thread");
        }

        pika::detail::runtime* rt = pika::detail::get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::suspend",
                "the runtime system is not active (did you already call pika::stop?)");
        }

        rt->suspend();
    }

    void resume()
    {
        if (threads::detail::get_self_ptr())
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::resume",
                "this function cannot be called from a pika thread");
        }

        pika::detail::runtime* rt = pika::detail::get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::resume",
                "the runtime system is not active (did you already call pika::stop?)");
        }

        rt->resume();
    }
}    // namespace pika
