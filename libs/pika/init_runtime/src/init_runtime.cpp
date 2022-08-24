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
#include <pika/execution/detail/execution_parameter_callbacks.hpp>
#include <pika/execution_base/detail/spinlock_deadlock_detection.hpp>
#include <pika/executors/exception_list.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/functional/function.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/init_runtime/detail/init_logging.hpp>
#include <pika/init_runtime/init_runtime.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/logging.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/timing.hpp>
#include <pika/parallel/util/detail/handle_exception_termination_handler.hpp>
#include <pika/program_options/parsers.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/resource_partitioner/partitioner.hpp>
#include <pika/runtime/config_entry.hpp>
#include <pika/runtime/custom_exception_info.hpp>
#include <pika/runtime/debugging.hpp>
#include <pika/runtime/get_locality_id.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/runtime_handlers.hpp>
#include <pika/runtime/shutdown_function.hpp>
#include <pika/runtime/startup_function.hpp>
#include <pika/string_util/classification.hpp>
#include <pika/string_util/split.hpp>
#include <pika/threading/thread.hpp>
#include <pika/type_support/pack.hpp>
#include <pika/type_support/unused.hpp>
#include <pika/util/from_string.hpp>
#include <pika/util/get_entry_as.hpp>
#include <pika/version.hpp>

#if defined(PIKA_HAVE_GPU_SUPPORT)
// TODO: Temporary, register this through global instead.
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#endif

#if defined(__bgq__)
#include <cstdlib>
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
#include <signal.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    namespace detail {

        int init_helper(pika::program_options::variables_map& /*vm*/,
            util::detail::function<int(int, char**)> const& f)
        {
            std::string cmdline(
                pika::get_config_entry("pika.reconstructed_cmd_line", ""));

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
    }    // namespace detail

    // Print stack trace and exit.
#if defined(PIKA_WINDOWS)
    extern BOOL WINAPI termination_handler(DWORD ctrl_type);
#else
    extern void termination_handler(int signum);
#endif

    int finalize(error_code& ec)
    {
        if (!is_running())
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::finalize",
                "the runtime system is not active (did you already "
                "call finalize?)");
            return -1;
        }

        if (&ec != &throws)
            ec = make_success_code();

        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::finalize",
                "the runtime system is not active (did you already "
                "call pika::stop?)");
            return -1;
        }

        rt->finalize(0);

        return 0;
    }

    int stop(error_code& ec)
    {
        if (threads::detail::get_self_ptr())
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::stop",
                "this function cannot be called from an pika thread");
            return -1;
        }

        std::unique_ptr<runtime> rt(get_runtime_ptr());    // take ownership!
        if (nullptr == rt.get())
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::stop",
                "the runtime system is not active (did you already "
                "call pika::stop?)");
            return -1;
        }

        int result = rt->wait();

        rt->stop();
        rt->rethrow_exception();

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    int suspend(error_code& ec)
    {
        if (threads::detail::get_self_ptr())
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::suspend",
                "this function cannot be called from an pika thread");
            return -1;
        }

        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::suspend",
                "the runtime system is not active (did you already "
                "call pika::stop?)");
            return -1;
        }

        return rt->suspend();
    }

    ///////////////////////////////////////////////////////////////////////////
    int resume(error_code& ec)
    {
        if (threads::detail::get_self_ptr())
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::resume",
                "this function cannot be called from an pika thread");
            return -1;
        }

        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROWS_IF(ec, pika::error::invalid_status, "pika::resume",
                "the runtime system is not active (did you already "
                "call pika::stop?)");
            return -1;
        }

        return rt->resume();
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        void activate_global_options(detail::command_line_handling& cmdline)
        {
#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__)
            threads::coroutines::detail::posix::use_guard_pages =
                cmdline.rtcfg_.use_stack_guard_pages();
#endif
#ifdef PIKA_HAVE_VERIFY_LOCKS
            if (cmdline.rtcfg_.enable_lock_detection())
            {
                util::enable_lock_detection();
                util::trace_depth_lock_detection(cmdline.rtcfg_.trace_depth());
            }
            else
            {
                util::disable_lock_detection();
            }
#endif
#ifdef PIKA_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
            threads::policies::set_minimal_deadlock_detection_enabled(
                cmdline.rtcfg_.enable_minimal_deadlock_detection());
#endif
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
            util::detail::set_spinlock_break_on_deadlock_enabled(
                cmdline.rtcfg_.enable_spinlock_deadlock_detection());
            util::detail::set_spinlock_deadlock_detection_limit(
                cmdline.rtcfg_.get_spinlock_deadlock_detection_limit());
            util::detail::set_spinlock_deadlock_warning_limit(
                cmdline.rtcfg_.get_spinlock_deadlock_warning_limit());
#endif
#if defined(PIKA_HAVE_LOGGING)
            util::detail::init_logging_local(cmdline.rtcfg_);
#else
            util::detail::warn_if_logging_requested(cmdline.rtcfg_);
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        void add_startup_functions(pika::runtime& rt,
            pika::program_options::variables_map& vm,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            if (vm.count("pika:app-config"))
            {
                std::string config(vm["pika:app-config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }

            if (!!startup)
                rt.add_startup_function(PIKA_MOVE(startup));

            if (!!shutdown)
                rt.add_shutdown_function(PIKA_MOVE(shutdown));

            if (vm.count("pika:dump-config-initial"))
            {
                std::cout << "Configuration after runtime construction:\n";
                std::cout << "-----------------------------------------\n";
                rt.get_config().dump(0, std::cout);
                std::cout << "-----------------------------------------\n";
            }

            if (vm.count("pika:dump-config"))
                rt.add_startup_function(dump_config(rt));
        }

        ///////////////////////////////////////////////////////////////////////
        int run(pika::runtime& rt,
            util::detail::function<int(
                pika::program_options::variables_map& vm)> const& f,
            pika::program_options::variables_map& vm,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            LPROGRESS_;

            add_startup_functions(
                rt, vm, PIKA_MOVE(startup), PIKA_MOVE(shutdown));

            // Run this runtime instance using the given function f.
            if (!f.empty())
                return rt.run(util::detail::bind_front(f, vm));

            // Run this runtime instance without an pika_main
            return rt.run();
        }

        int start(pika::runtime& rt,
            util::detail::function<int(
                pika::program_options::variables_map& vm)> const& f,
            pika::program_options::variables_map& vm,
            startup_function_type startup, shutdown_function_type shutdown)
        {
            LPROGRESS_;

            add_startup_functions(
                rt, vm, PIKA_MOVE(startup), PIKA_MOVE(shutdown));

            if (!f.empty())
            {
                // Run this runtime instance using the given function f.
                return rt.start(util::detail::bind_front(f, vm));
            }

            // Run this runtime instance without an pika_main
            return rt.start();
        }

        int run_or_start(bool blocking, std::unique_ptr<pika::runtime> rt,
            detail::command_line_handling& cfg, startup_function_type startup,
            shutdown_function_type shutdown)
        {
#if defined(PIKA_HAVE_GPU_SUPPORT)
            pika::register_pre_startup_function(
                pika::cuda::experimental::detail::init_polling_thread);
            pika::register_shutdown_function(
                pika::cuda::experimental::detail::finalize_polling_thread);
#endif

            if (blocking)
            {
                return run(*rt, cfg.pika_main_f_, cfg.vm_, PIKA_MOVE(startup),
                    PIKA_MOVE(shutdown));
            }

            // non-blocking version
            start(*rt, cfg.pika_main_f_, cfg.vm_, PIKA_MOVE(startup),
                PIKA_MOVE(shutdown));

            // pointer to runtime is stored in TLS
            pika::runtime* p = rt.release();
            (void) p;

            return 0;
        }

        ////////////////////////////////////////////////////////////////////////
        void init_environment()
        {
            PIKA_UNUSED(pika::detail::filesystem::initial_path());

            pika::detail::set_assertion_handler(
                &pika::detail::assertion_handler);
            pika::detail::set_custom_exception_info_handler(
                &pika::detail::custom_exception_info);
            pika::detail::set_pre_exception_handler(
                &pika::detail::pre_exception_handler);
            pika::set_thread_termination_handler(
                [](std::exception_ptr const& e) { report_error(e); });
            pika::lcos::detail::set_run_on_completed_error_handler(
                [](std::exception_ptr const& e) {
                    pika::detail::report_exception_and_terminate(e);
                });
            pika::detail::set_get_full_build_string(&pika::full_build_string);
#if defined(PIKA_HAVE_VERIFY_LOCKS)
            pika::util::set_registered_locks_error_handler(
                &pika::detail::registered_locks_error_handler);
            pika::util::set_register_locks_predicate(
                &pika::detail::register_locks_predicate);
#endif
#if !defined(PIKA_HAVE_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
            set_error_handlers();
#endif
            pika::threads::detail::set_get_default_pool(
                &pika::detail::get_default_pool);
            pika::threads::detail::set_get_locality_id(&get_locality_id);
            pika::parallel::execution::detail::set_get_pu_mask(
                &pika::detail::get_pu_mask);
            pika::parallel::execution::detail::set_get_os_thread_count(
                []() { return pika::get_os_thread_count(); });

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

        // make sure the runtime system is not active yet
        int ensure_no_runtime_is_up()
        {
            // make sure the runtime system is not active yet
            if (get_runtime_ptr() != nullptr)
            {
                std::cerr << "pika::init: can't initialize runtime system "
                             "more than once! Exiting...\n";
                return -1;
            }
            return 0;
        }

        ///////////////////////////////////////////////////////////////////////
        int run_or_start(
            util::detail::function<int(
                pika::program_options::variables_map& vm)> const& f,
            int argc, const char* const* argv, init_params const& params,
            bool blocking)
        {
            init_environment();

            int result = 0;
            try
            {
                if ((result = ensure_no_runtime_is_up()) != 0)
                {
                    return result;
                }

                pika::detail::command_line_handling cmdline{
                    pika::util::runtime_configuration(
                        argv[0], pika::runtime_mode::local),
                    params.cfg, f};

                // scope exception handling to resource partitioner initialization
                // any exception thrown during run_or_start below are handled
                // separately
                try
                {
                    result = cmdline.call(params.desc_cmdline, argc, argv);

                    pika::detail::affinity_data affinity_data{};
                    affinity_data.init(
                        pika::util::get_entry_as<std::size_t>(
                            cmdline.rtcfg_, "pika.os_threads", 0),
                        pika::util::get_entry_as<std::size_t>(
                            cmdline.rtcfg_, "pika.cores", 0),
                        pika::util::get_entry_as<std::size_t>(
                            cmdline.rtcfg_, "pika.pu_offset", 0),
                        pika::util::get_entry_as<std::size_t>(
                            cmdline.rtcfg_, "pika.pu_step", 0),
                        0, cmdline.rtcfg_.get_entry("pika.affinity", ""),
                        cmdline.rtcfg_.get_entry("pika.bind", ""),
                        !pika::util::get_entry_as<bool>(
                            cmdline.rtcfg_, "pika.ignore_process_mask", false));

                    pika::resource::partitioner rp =
                        pika::resource::detail::make_partitioner(
                            params.rp_mode, cmdline.rtcfg_, affinity_data);

                    activate_global_options(cmdline);

                    // check whether pika should be exited at this point
                    // (parse_result is returning a result > 0, if the program options
                    // contain --pika:help or --pika:version, on error result is < 0)
                    if (result != 0)
                    {
                        if (result > 0)
                            result = 0;
                        return result;
                    }

                    // If thread_pools initialization in user main
                    if (params.rp_callback)
                    {
                        params.rp_callback(rp, cmdline.vm_);
                    }

                    // Setup all internal parameters of the resource_partitioner
                    rp.configure_pools();
                }
                catch (pika::exception const& e)
                {
                    std::cerr << "pika::init: pika::exception caught: "
                              << pika::get_error_what(e) << "\n";
                    return -1;
                }

                // Initialize and start the pika runtime.
                LPROGRESS_ << "run_local: create runtime";

                // Build and configure this runtime instance.
                std::unique_ptr<pika::runtime> rt;

                // Command line handling should have updated this by now.
                LPROGRESS_ << "creating local runtime";
                rt.reset(new pika::runtime(cmdline.rtcfg_, true));

                result = run_or_start(blocking, PIKA_MOVE(rt), cmdline,
                    PIKA_MOVE(params.startup), PIKA_MOVE(params.shutdown));
            }
            catch (pika::detail::command_line_error const& e)
            {
                std::cerr << "pika::init: std::exception caught: " << e.what()
                          << "\n";
                return -1;
            }
            return result;
        }
    }    // namespace detail
}    // namespace pika
