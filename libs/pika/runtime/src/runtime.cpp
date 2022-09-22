//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/command_line_handling/late_command_line_handling.hpp>
#include <pika/command_line_handling/parse_command_line.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/debugging/backtrace.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/functional/bind.hpp>
#include <pika/functional/function.hpp>
#include <pika/itt_notify/thread_name.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/logging.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/runtime/config_entry.hpp>
#include <pika/runtime/custom_exception_info.hpp>
#include <pika/runtime/debugging.hpp>
#include <pika/runtime/os_thread_type.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/runtime_fwd.hpp>
#include <pika/runtime/shutdown_function.hpp>
#include <pika/runtime/startup_function.hpp>
#include <pika/runtime/state.hpp>
#include <pika/runtime/thread_hooks.hpp>
#include <pika/runtime/thread_mapper.hpp>
#include <pika/thread_support/set_thread_name.hpp>
#include <pika/threading_base/external_timer.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/topology/topology.hpp>
#include <pika/util/from_string.hpp>
#include <pika/version.hpp>

#if defined(PIKA_HAVE_TRACY)
#include <common/TracySystem.hpp>
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Make sure the system gets properly shut down while handling Ctrl-C and other
// system signals
#if defined(PIKA_WINDOWS)

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    void handle_termination(char const* reason)
    {
        if (get_config_entry("pika.attach_debugger", "") == "exception")
        {
            debug::detail::attach_debugger();
        }

        if (get_config_entry("pika.diagnostics_on_terminate", "1") == "1")
        {
            int const verbosity = util::from_string<int>(
                get_config_entry("pika.exception_verbosity", "1"));

            if (verbosity >= 2)
            {
                std::cerr << pika::full_build_string() << "\n";
            }

#if defined(PIKA_HAVE_STACKTRACES)
            if (verbosity >= 1)
            {
                std::size_t const trace_depth =
                    util::from_string<std::size_t>(get_config_entry(
                        "pika.trace_depth", PIKA_HAVE_THREAD_BACKTRACE_DEPTH));
                std::cerr << "{stack-trace}: "
                          << pika::debug::detail::trace(trace_depth) << "\n";
            }
#endif

            std::cerr << "{what}: " << (reason ? reason : "Unknown reason")
                      << "\n";
        }
    }

    PIKA_EXPORT BOOL WINAPI termination_handler(DWORD ctrl_type)
    {
        switch (ctrl_type)
        {
        case CTRL_C_EVENT:
            handle_termination("Ctrl-C");
            return TRUE;

        case CTRL_BREAK_EVENT:
            handle_termination("Ctrl-Break");
            return TRUE;

        case CTRL_CLOSE_EVENT:
            handle_termination("Ctrl-Close");
            return TRUE;

        case CTRL_LOGOFF_EVENT:
            handle_termination("Logoff");
            return TRUE;

        case CTRL_SHUTDOWN_EVENT:
            handle_termination("Shutdown");
            return TRUE;

        default:
            break;
        }
        return FALSE;
    }
}    // namespace pika

#else

#include <signal.h>
#include <stdlib.h>
#include <string.h>

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    [[noreturn]] PIKA_EXPORT void termination_handler(int signum)
    {
        if (signum != SIGINT &&
            get_config_entry("pika.attach_debugger", "") == "exception")
        {
            debug::detail::attach_debugger();
        }

        if (get_config_entry("pika.diagnostics_on_terminate", "1") == "1")
        {
            int const verbosity = util::from_string<int>(
                get_config_entry("pika.exception_verbosity", "1"));

            char* reason = strsignal(signum);

            if (verbosity >= 2)
            {
                std::cerr << pika::full_build_string() << "\n";
            }

#if defined(PIKA_HAVE_STACKTRACES)
            if (verbosity >= 1)
            {
                std::size_t const trace_depth =
                    util::from_string<std::size_t>(get_config_entry(
                        "pika.trace_depth", PIKA_HAVE_THREAD_BACKTRACE_DEPTH));
                std::cerr << "{stack-trace}: "
                          << pika::debug::detail::trace(trace_depth) << "\n";
            }
#endif

            std::cerr << "{what}: " << (reason ? reason : "Unknown reason")
                      << "\n";
        }
        std::abort();
    }
}    // namespace pika

#endif

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT void PIKA_CDECL new_handler()
    {
        PIKA_THROW_EXCEPTION(pika::error::out_of_memory, "new_handler",
            "new allocator failed to allocate memory");
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // Sometimes the pika library gets simply unloaded as a result of some
        // extreme error handling. Avoid hangs in the end by setting a flag.
        static bool exit_called = false;

        void on_exit() noexcept
        {
            exit_called = true;
        }

        void on_abort(int) noexcept
        {
            exit_called = true;
            std::exit(-1);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    void set_signal_handlers()
    {
#if defined(PIKA_WINDOWS)
        // Set console control handler to allow server to be stopped.
        SetConsoleCtrlHandler(pika::termination_handler, TRUE);
#else
        struct sigaction new_action;
        new_action.sa_handler = pika::termination_handler;
        sigemptyset(&new_action.sa_mask);
        new_action.sa_flags = 0;

        sigaction(SIGINT, &new_action, nullptr);     // Interrupted
        sigaction(SIGBUS, &new_action, nullptr);     // Bus error
        sigaction(SIGFPE, &new_action, nullptr);     // Floating point exception
        sigaction(SIGILL, &new_action, nullptr);     // Illegal instruction
        sigaction(SIGPIPE, &new_action, nullptr);    // Bad pipe
        sigaction(SIGSEGV, &new_action, nullptr);    // Segmentation fault
        sigaction(SIGSYS, &new_action, nullptr);     // Bad syscall
#endif

        std::set_new_handler(pika::new_handler);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace strings {
        char const* const runtime_state_names[] = {
            "runtime_state::invalid",         // -1
            "runtime_state::initialized",     // 0
            "runtime_state::pre_startup",     // 1
            "runtime_state::startup",         // 2
            "runtime_state::pre_main",        // 3
            "runtime_state::starting",        // 4
            "runtime_state::running",         // 5
            "runtime_state::suspended",       // 6
            "runtime_state::pre_sleep",       // 7
            "runtime_state::sleeping",        // 8
            "runtime_state::pre_shutdown",    // 9
            "runtime_state::shutdown",        // 10
            "runtime_state::stopping",        // 11
            "runtime_state::terminating",     // 12
            "runtime_state::stopped"          // 13
        };
    }

    namespace detail {
        char const* get_runtime_state_name(runtime_state st)
        {
            if (st < runtime_state::invalid ||
                st >= runtime_state::last_valid_runtime)
                return "invalid (value out of bounds)";
            return strings::runtime_state_names[static_cast<std::int8_t>(st) +
                1];
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    threads::callback_notifier::on_startstop_type global_on_start_func;
    threads::callback_notifier::on_startstop_type global_on_stop_func;
    threads::callback_notifier::on_error_type global_on_error_func;

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(pika::util::runtime_configuration& rtcfg, bool initialize)
      : rtcfg_(rtcfg)
      , instance_number_(++instance_number_counter_)
      , thread_support_(new util::thread_mapper)
      , topology_(resource::get_partitioner().get_topology())
      , state_(runtime_state::invalid)
      , on_start_func_(global_on_start_func)
      , on_stop_func_(global_on_stop_func)
      , on_error_func_(global_on_error_func)
      , result_(0)
      , notifier_()
      , thread_manager_()
      , stop_called_(false)
      , stop_done_(false)
    {
        LPROGRESS_;

        // set notification policies only after the object was completely
        // initialized
        runtime::set_notification_policies(
            runtime::get_notification_policy(
                "worker-thread", os_thread_type::worker_thread),
            threads::detail::network_background_callback_type{});

        init_global_data();

        if (initialize)
        {
            init();
        }
    }

    // this constructor is called by the distributed runtime only
    runtime::runtime(pika::util::runtime_configuration& rtcfg)
      : rtcfg_(rtcfg)
      , instance_number_(++instance_number_counter_)
      , thread_support_(new util::thread_mapper)
      , topology_(resource::get_partitioner().get_topology())
      , state_(runtime_state::invalid)
      , on_start_func_(global_on_start_func)
      , on_stop_func_(global_on_stop_func)
      , on_error_func_(global_on_error_func)
      , result_(0)
      , notifier_()
      , thread_manager_()
      , stop_called_(false)
      , stop_done_(false)
    {
        init_global_data();

        LPROGRESS_;
    }

    void runtime::set_notification_policies(notification_policy_type&& notifier,
        threads::detail::network_background_callback_type
            network_background_callback)
    {
        notifier_ = PIKA_MOVE(notifier);

        thread_manager_.reset(new pika::threads::detail::thread_manager(
            rtcfg_, notifier_, network_background_callback));
    }

    void runtime::init()
    {
        LPROGRESS_;

        try
        {
            // now create all thread_manager pools
            thread_manager_->create_pools();

            // this initializes the used_processing_units_ mask
            thread_manager_->init();

            // copy over all startup functions registered so far
            for (startup_function_type& f :
                detail::global_pre_startup_functions)
            {
                add_pre_startup_function(f);
            }

            for (startup_function_type& f : detail::global_startup_functions)
            {
                add_startup_function(f);
            }

            for (shutdown_function_type& f :
                detail::global_pre_shutdown_functions)
            {
                add_pre_shutdown_function(f);
            }

            for (shutdown_function_type& f : detail::global_shutdown_functions)
            {
                add_shutdown_function(f);
            }
        }
        catch (std::exception const& e)
        {
            // errors at this point need to be reported directly
            detail::report_exception_and_terminate(e);
        }
        catch (...)
        {
            // errors at this point need to be reported directly
            detail::report_exception_and_terminate(std::current_exception());
        }

        // set state to initialized
        set_state(runtime_state::initialized);
    }

    runtime::~runtime()
    {
        LRT_(debug).format("~runtime(entering)");

        // stop all services
        thread_manager_->stop();
        LRT_(debug).format("~runtime(finished)");

        LPROGRESS_;

        // allow to reuse instance number if this was the only instance
        if (0 == instance_number_counter_)
            --instance_number_counter_;

        resource::detail::delete_partitioner();
    }

    void runtime::on_exit(util::detail::function<void()> const& f)
    {
        std::lock_guard<std::mutex> l(mtx_);
        on_exit_functions_.push_back(f);
    }

    void runtime::starting()
    {
        state_.store(runtime_state::pre_main);
    }

    void runtime::stopping()
    {
        state_.store(runtime_state::stopped);

        using value_type = util::detail::function<void()>;

        std::lock_guard<std::mutex> l(mtx_);
        for (value_type const& f : on_exit_functions_)
            f();
    }

    bool runtime::stopped() const
    {
        return state_.load() == runtime_state::stopped;
    }

    pika::util::runtime_configuration& runtime::get_config()
    {
        return rtcfg_;
    }

    pika::util::runtime_configuration const& runtime::get_config() const
    {
        return rtcfg_;
    }

    std::size_t runtime::get_instance_number() const
    {
        return static_cast<std::size_t>(instance_number_);
    }

    runtime_state runtime::get_state() const
    {
        return state_.load();
    }

    threads::detail::topology const& runtime::get_topology() const
    {
        return topology_;
    }

    void runtime::set_state(runtime_state s)
    {
        LPROGRESS_ << pika::detail::get_runtime_state_name(s);
        state_.store(s);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::atomic<int> runtime::instance_number_counter_(-1);

    ///////////////////////////////////////////////////////////////////////////
    namespace {
        std::chrono::time_point<std::chrono::high_resolution_clock>&
        runtime_uptime()
        {
            static std::chrono::time_point<std::chrono::high_resolution_clock>
                uptime = std::chrono::high_resolution_clock::now();
            return uptime;
        }
    }    // namespace

    void runtime::init_global_data()
    {
        runtime*& runtime_ = get_runtime_ptr();
        PIKA_ASSERT(!runtime_);
        PIKA_ASSERT(nullptr == threads::detail::thread_self::get_self());

        runtime_ = this;
        runtime_uptime() = std::chrono::high_resolution_clock::now();
    }

    void runtime::deinit_global_data()
    {
        runtime*& runtime_ = get_runtime_ptr();
        PIKA_ASSERT(runtime_);
        runtime_ = nullptr;
    }

    std::uint64_t runtime::get_system_uptime()
    {
        using std::chrono::high_resolution_clock;
        auto diff = (high_resolution_clock::now() - runtime_uptime()).count();
        return diff < 0LL ? 0ULL : static_cast<std::uint64_t>(diff);
    }

    threads::callback_notifier::on_startstop_type runtime::on_start_func() const
    {
        return on_start_func_;
    }

    threads::callback_notifier::on_startstop_type runtime::on_stop_func() const
    {
        return on_stop_func_;
    }

    threads::callback_notifier::on_error_type runtime::on_error_func() const
    {
        return on_error_func_;
    }

    threads::callback_notifier::on_startstop_type runtime::on_start_func(
        threads::callback_notifier::on_startstop_type&& f)
    {
        threads::callback_notifier::on_startstop_type newf = PIKA_MOVE(f);
        std::swap(on_start_func_, newf);
        return newf;
    }

    threads::callback_notifier::on_startstop_type runtime::on_stop_func(
        threads::callback_notifier::on_startstop_type&& f)
    {
        threads::callback_notifier::on_startstop_type newf = PIKA_MOVE(f);
        std::swap(on_stop_func_, newf);
        return newf;
    }

    threads::callback_notifier::on_error_type runtime::on_error_func(
        threads::callback_notifier::on_error_type&& f)
    {
        threads::callback_notifier::on_error_type newf = PIKA_MOVE(f);
        std::swap(on_error_func_, newf);
        return newf;
    }

    std::uint32_t runtime::get_locality_id(error_code& /* ec */) const
    {
        return 0;
    }

    std::size_t runtime::get_num_worker_threads() const
    {
        PIKA_ASSERT(thread_manager_);
        return thread_manager_->get_os_thread_count();
    }

    std::uint32_t runtime::get_num_localities(
        pika::launch::sync_policy, error_code& /* ec */) const
    {
        return 1;
    }

    std::uint32_t runtime::get_initial_num_localities() const
    {
        return 1;
    }

    pika::future<std::uint32_t> runtime::get_num_localities() const
    {
        return make_ready_future(std::uint32_t(1));
    }

    std::string runtime::get_locality_name() const
    {
        return "console";
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::callback_notifier::on_startstop_type get_thread_on_start_func()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_start_func();
        }
        else
        {
            return global_on_start_func;
        }
    }

    threads::callback_notifier::on_startstop_type get_thread_on_stop_func()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_stop_func();
        }
        else
        {
            return global_on_stop_func;
        }
    }

    threads::callback_notifier::on_error_type get_thread_on_error_func()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_error_func();
        }
        else
        {
            return global_on_error_func;
        }
    }

    threads::callback_notifier::on_startstop_type register_thread_on_start_func(
        threads::callback_notifier::on_startstop_type&& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_start_func(PIKA_MOVE(f));
        }

        threads::callback_notifier::on_startstop_type newf = PIKA_MOVE(f);
        std::swap(global_on_start_func, newf);
        return newf;
    }

    threads::callback_notifier::on_startstop_type register_thread_on_stop_func(
        threads::callback_notifier::on_startstop_type&& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_stop_func(PIKA_MOVE(f));
        }

        threads::callback_notifier::on_startstop_type newf = PIKA_MOVE(f);
        std::swap(global_on_stop_func, newf);
        return newf;
    }

    threads::callback_notifier::on_error_type register_thread_on_error_func(
        threads::callback_notifier::on_error_type&& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_error_func(PIKA_MOVE(f));
        }

        threads::callback_notifier::on_error_type newf = PIKA_MOVE(f);
        std::swap(global_on_error_func, newf);
        return newf;
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime& get_runtime()
    {
        PIKA_ASSERT(get_runtime_ptr() != nullptr);
        return *get_runtime_ptr();
    }

    runtime*& get_runtime_ptr()
    {
        static runtime* runtime_ = nullptr;
        return runtime_;
    }

    std::string get_thread_name()
    {
        std::string& thread_name = detail::thread_name();
        if (thread_name.empty())
            return "<unknown>";
        return thread_name;
    }

    // Register the current kernel thread with pika, this should be done once
    // for each external OS-thread intended to invoke pika functionality.
    // Calling this function more than once will silently fail
    // (will return false).
    bool register_thread(runtime* rt, char const* name, error_code& ec)
    {
        PIKA_ASSERT(rt);
        return rt->register_thread(name, 0, true, ec);
    }

    // Unregister the thread from pika, this should be done once in
    // the end before the external thread exists.
    void unregister_thread(runtime* rt)
    {
        PIKA_ASSERT(rt);
        rt->unregister_thread();
    }

    // Access data for a given OS thread that was previously registered by
    // \a register_thread. This function must be called from a thread that was
    // previously registered with the runtime.
    os_thread_data get_os_thread_data(std::string const& label)
    {
        return get_runtime().get_os_thread_data(label);
    }

    /// Enumerate all OS threads that have registered with the runtime.
    bool enumerate_os_threads(
        util::detail::function<bool(os_thread_data const&)> const& f)
    {
        return get_runtime().enumerate_os_threads(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    void report_error(std::size_t num_thread, std::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::thread_manager_is(runtime_state::running))
        {
            pika::runtime* rt = pika::get_runtime_ptr();
            if (rt)
                rt->report_error(num_thread, e);
            else
                detail::report_exception_and_terminate(e);
            return;
        }

        get_runtime().get_thread_manager().report_error(num_thread, e);
    }

    void report_error(std::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::thread_manager_is(runtime_state::running))
        {
            pika::runtime* rt = pika::get_runtime_ptr();
            if (rt)
                rt->report_error(std::size_t(-1), e);
            else
                detail::report_exception_and_terminate(e);
            return;
        }

        std::size_t num_thread = pika::get_worker_thread_num();
        get_runtime().get_thread_manager().report_error(num_thread, e);
    }

    bool register_on_exit(util::detail::function<void()> const& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
            return false;

        rt->on_exit(f);
        return true;
    }

    std::size_t get_runtime_instance_number()
    {
        runtime* rt = get_runtime_ptr();
        return (nullptr == rt) ? 0 : rt->get_instance_number();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string get_config_entry(
        std::string const& key, std::string const& dflt)
    {
        if (get_runtime_ptr() != nullptr)
        {
            return get_runtime().get_config().get_entry(key, dflt);
        }

        return dflt;
    }

    std::string get_config_entry(std::string const& key, std::size_t dflt)
    {
        if (get_runtime_ptr() != nullptr)
        {
            return get_runtime().get_config().get_entry(key, dflt);
        }

        return std::to_string(dflt);
    }

    // set entries
    void set_config_entry(std::string const& key, std::string const& value)
    {
        if (get_runtime_ptr() != nullptr)
        {
            get_runtime_ptr()->get_config().add_entry(key, value);
            return;
        }
    }

    void set_config_entry(std::string const& key, std::size_t value)
    {
        set_config_entry(key, std::to_string(value));
    }

    void set_config_entry_callback(std::string const& key,
        util::detail::function<void(
            std::string const&, std::string const&)> const& callback)
    {
        if (get_runtime_ptr() != nullptr)
        {
            get_runtime_ptr()->get_config().add_notification_callback(
                key, callback);
            return;
        }
    }

    namespace detail {
        ///////////////////////////////////////////////////////////////////////////
        // retrieve the command line arguments for the current locality
        bool retrieve_commandline_arguments(
            pika::program_options::options_description const& app_options,
            pika::program_options::variables_map& vm)
        {
            // The command line for this application instance is available from
            // this configuration section:
            //
            //     [pika]
            //     cmd_line=....
            //
            std::string cmdline;

            pika::util::section& cfg = pika::get_runtime().get_config();
            if (cfg.has_entry("pika.cmd_line"))
                cmdline = cfg.get_entry("pika.cmd_line");

            return parse_commandline(cfg, app_options, cmdline, vm,
                commandline_error_mode::allow_unregistered);
        }

        ///////////////////////////////////////////////////////////////////////////
        // retrieve the command line arguments for the current locality
        bool retrieve_commandline_arguments(std::string const& appname,
            pika::program_options::variables_map& vm)
        {
            using pika::program_options::options_description;

            options_description desc_commandline(
                "Usage: " + appname + " [options]");

            return retrieve_commandline_arguments(desc_commandline, vm);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_os_thread_count()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "pika::get_os_thread_count()",
                "the runtime system has not been initialized yet");
            return std::size_t(0);
        }
        return rt->get_config().get_os_thread_count();
    }

    bool is_scheduler_numa_sensitive()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "pika::is_scheduler_numa_sensitive",
                "the runtime system has not been initialized yet");
            return false;
        }

        if (std::size_t(-1) != get_worker_thread_num())
            return false;
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool is_running()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
            return rt->get_state() == runtime_state::running;
        return false;
    }

    bool is_stopped()
    {
        if (!detail::exit_called)
        {
            runtime* rt = get_runtime_ptr();
            if (nullptr != rt)
                return rt->get_state() == runtime_state::stopped;
        }
        return true;    // assume stopped
    }

    bool is_stopped_or_shutting_down()
    {
        runtime* rt = get_runtime_ptr();
        if (!detail::exit_called && nullptr != rt)
        {
            runtime_state st = rt->get_state();
            return st >= runtime_state::shutdown;
        }
        return true;    // assume stopped
    }

    bool tolerate_node_faults()
    {
#ifdef PIKA_HAVE_FAULT_TOLERANCE
        return true;
#else
        return false;
#endif
    }

    bool is_starting()
    {
        runtime* rt = get_runtime_ptr();
        return nullptr != rt ? rt->get_state() <= runtime_state::startup : true;
    }

    bool is_pre_startup()
    {
        runtime* rt = get_runtime_ptr();
        return nullptr != rt ? rt->get_state() < runtime_state::startup : true;
    }
}    // namespace pika

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {
    std::string expand(std::string const& in)
    {
        return get_runtime().get_config().expand(in);
    }

    void expand(std::string& in)
    {
        get_runtime().get_config().expand(in, std::string::size_type(-1));
    }
}}    // namespace pika::util

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace threads {
    detail::thread_manager& get_thread_manager()
    {
        return get_runtime().get_thread_manager();
    }

    // shortcut for runtime_configuration::get_default_stack_size
    std::ptrdiff_t get_default_stack_size()
    {
        return get_runtime().get_config().get_default_stack_size();
    }

    // shortcut for runtime_configuration::get_stack_size
    std::ptrdiff_t get_stack_size(execution::thread_stacksize stacksize)
    {
        if (stacksize == execution::thread_stacksize::current)
            return threads::detail::get_self_stacksize();

        return get_runtime().get_config().get_stack_size(stacksize);
    }

    void reset_thread_distribution()
    {
        get_runtime().get_thread_manager().reset_thread_distribution();
    }

    void set_scheduler_mode(threads::scheduler_mode m)
    {
        get_runtime().get_thread_manager().set_scheduler_mode(m);
    }

    void add_scheduler_mode(threads::scheduler_mode m)
    {
        get_runtime().get_thread_manager().add_scheduler_mode(m);
    }

    void add_remove_scheduler_mode(threads::scheduler_mode to_add_mode,
        threads::scheduler_mode to_remove_mode)
    {
        get_runtime().get_thread_manager().add_remove_scheduler_mode(
            to_add_mode, to_remove_mode);
    }

    void remove_scheduler_mode(threads::scheduler_mode m)
    {
        get_runtime().get_thread_manager().remove_scheduler_mode(m);
    }

    detail::topology const& get_topology()
    {
        pika::runtime* rt = pika::get_runtime_ptr();
        if (rt == nullptr)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "pika::threads::get_topology",
                "the pika runtime system has not been initialized yet");
        }
        return rt->get_topology();
    }
}}    // namespace pika::threads

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    std::uint64_t get_system_uptime()
    {
        return runtime::get_system_uptime();
    }

    pika::util::runtime_configuration const& get_config()
    {
        return get_runtime().get_config();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Return true if networking is enabled.
    bool is_networking_enabled()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->is_networking_enabled();
        }
        return true;    // be on the safe side, enable networking
    }
}    // namespace pika

#if defined(_WIN64) && defined(PIKA_DEBUG) &&                                  \
    !defined(PIKA_HAVE_FIBER_BASED_COROUTINES)
#include <io.h>
#endif

namespace pika {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // There is no need to protect these global from thread concurrent
        // access as they are access during early startup only.
        std::list<startup_function_type> global_pre_startup_functions;
        std::list<startup_function_type> global_startup_functions;
        std::list<shutdown_function_type> global_pre_shutdown_functions;
        std::list<shutdown_function_type> global_shutdown_functions;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    void register_pre_startup_function(startup_function_type f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            if (rt->get_state() > runtime_state::pre_startup)
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                    "register_pre_startup_function",
                    "Too late to register a new pre-startup function.");
                return;
            }
            rt->add_pre_startup_function(PIKA_MOVE(f));
        }
        else
        {
            detail::global_pre_startup_functions.push_back(PIKA_MOVE(f));
        }
    }

    void register_startup_function(startup_function_type f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            if (rt->get_state() > runtime_state::startup)
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                    "register_startup_function",
                    "Too late to register a new startup function.");
                return;
            }
            rt->add_startup_function(PIKA_MOVE(f));
        }
        else
        {
            detail::global_startup_functions.push_back(PIKA_MOVE(f));
        }
    }

    void register_pre_shutdown_function(shutdown_function_type f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            if (rt->get_state() > runtime_state::pre_shutdown)
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                    "register_pre_shutdown_function",
                    "Too late to register a new pre-shutdown function.");
                return;
            }
            rt->add_pre_shutdown_function(PIKA_MOVE(f));
        }
        else
        {
            detail::global_pre_shutdown_functions.push_back(PIKA_MOVE(f));
        }
    }

    void register_shutdown_function(shutdown_function_type f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            if (rt->get_state() > runtime_state::shutdown)
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                    "register_shutdown_function",
                    "Too late to register a new shutdown function.");
                return;
            }
            rt->add_shutdown_function(PIKA_MOVE(f));
        }
        else
        {
            detail::global_shutdown_functions.push_back(PIKA_MOVE(f));
        }
    }

    void runtime::call_startup_functions(bool pre_startup)
    {
        if (pre_startup)
        {
            set_state(runtime_state::pre_startup);
            for (startup_function_type& f : pre_startup_functions_)
            {
                f();
            }
        }
        else
        {
            set_state(runtime_state::startup);
            for (startup_function_type& f : startup_functions_)
            {
                f();
            }
        }
    }

    void runtime::call_shutdown_functions(bool pre_shutdown)
    {
        if (pre_shutdown)
        {
            for (shutdown_function_type& f : pre_shutdown_functions_)
            {
                f();
            }
        }
        else
        {
            for (shutdown_function_type& f : shutdown_functions_)
            {
                f();
            }
        }
    }

    namespace detail {
        void handle_print_bind(std::size_t num_threads)
        {
            threads::detail::topology& top = threads::detail::create_topology();
            auto const& rp = pika::resource::get_partitioner();
            auto const& tm = get_runtime().get_thread_manager();

            {
                // make sure all output is kept together
                std::ostringstream strm;

                strm << std::string(79, '*') << '\n';
                strm << "locality: " << pika::get_locality_id() << '\n';
                for (std::size_t i = 0; i != num_threads; ++i)
                {
                    // print the mask for the current PU
                    threads::detail::mask_cref_type pu_mask = rp.get_pu_mask(i);

                    if (!threads::detail::any(pu_mask))
                    {
                        strm << std::setw(4) << i    //-V112
                             << ": thread binding disabled" << std::endl;
                    }
                    else
                    {
                        std::string pool_name = tm.get_pool(i).get_pool_name();
                        top.print_affinity_mask(strm, i, pu_mask, pool_name);
                    }

                    // Make sure the mask does not contradict the CPU bindings
                    // returned by the system (see #973: Would like option to
                    // report HWLOC bindings).
                    error_code ec(throwmode::lightweight);
                    std::thread& blob = tm.get_os_thread_handle(i);
                    threads::detail::mask_type boundcpu =
                        top.get_cpubind_mask(blob, ec);

                    // The masks reported by pika must be the same as the ones
                    // reported from HWLOC.
                    if (!ec && threads::detail::any(boundcpu) &&
                        !threads::detail::equal(boundcpu, pu_mask, num_threads))
                    {
                        std::string boundcpu_str =
                            threads::detail::to_string(boundcpu);
                        std::string pu_mask_str =
                            threads::detail::to_string(pu_mask);
                        PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                            "handle_print_bind",
                            pika::util::format(
                                "unexpected mismatch between locality {1}: "
                                "binding reported from HWLOC({2}) and "
                                "pika({3}) "
                                "on thread {4}",
                                pika::get_locality_id(), boundcpu_str,
                                pu_mask_str, i));
                    }
                }

                std::cout << strm.str();
            }
        }
    }    // namespace detail

    threads::detail::thread_result_type runtime::run_helper(
        util::detail::function<runtime::pika_main_function_type> const& func,
        int& result, bool call_startup)
    {
        bool caught_exception = false;
        try
        {
            {
                pika::program_options::options_description options;
                result = pika::detail::handle_late_commandline_options(
                    get_config(), options, &detail::handle_print_bind);
                if (result)
                {
                    lbt_ << "runtime::run_helper: bootstrap "
                            "aborted, bailing out";

                    set_state(runtime_state::running);
                    finalize(-1.0);

                    return threads::detail::thread_result_type(
                        threads::detail::thread_schedule_state::terminated,
                        threads::detail::invalid_thread_id);
                }
            }

            if (call_startup)
            {
                call_startup_functions(true);
                lbt_ << "(3rd stage) run_helper: ran pre-startup functions";

                call_startup_functions(false);
                lbt_ << "(4th stage) run_helper: ran startup functions";
            }

            lbt_ << "(4th stage) runtime::run_helper: bootstrap complete";
            set_state(runtime_state::running);

            // Now, execute the user supplied thread function (pika_main)
            if (!!func)
            {
                lbt_ << "(last stage) runtime::run_helper: about to "
                        "invoke pika_main";

                // Change our thread description, as we're about to call pika_main
                threads::detail::set_thread_description(
                    threads::detail::get_self_id(), "pika_main");

                // Call pika_main
                result = func();
            }
        }
        catch (...)
        {
            // make sure exceptions thrown in pika_main don't escape
            // unnoticed
            {
                std::lock_guard<std::mutex> l(mtx_);
                exception_ = std::current_exception();
            }
            result = -1;
            caught_exception = true;
        }

        if (caught_exception)
        {
            PIKA_ASSERT(exception_);
            report_error(exception_, false);
            finalize(-1.0);    // make sure the application exits
        }

        return threads::detail::thread_result_type(
            threads::detail::thread_schedule_state::terminated,
            threads::detail::invalid_thread_id);
    }

    int runtime::start(
        util::detail::function<pika_main_function_type> const& func,
        bool blocking)
    {
#if defined(_WIN64) && defined(PIKA_DEBUG) &&                                  \
    !defined(PIKA_HAVE_FIBER_BASED_COROUTINES)
        // needs to be called to avoid problems at system startup
        // see: http://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=100319
        _isatty(0);
#endif
        // {{{ early startup code - local

        // initialize instrumentation system
#ifdef PIKA_HAVE_APEX
        detail::external_timer::init(nullptr, 0, 1);
#endif

        LRT_(info).format("cmd_line: {}", get_config().get_cmd_line());

        lbt_ << "(1st stage) runtime::start: booting locality " << here();

        // Register this thread with the runtime system to allow calling
        // certain pika functionality from the main thread. Also calls
        // registered startup callbacks.
        init_tss_helper(
            "main-thread", os_thread_type::main_thread, 0, 0, "", "", false);

        // start the thread manager
        thread_manager_->run();
        lbt_ << "(1st stage) runtime::start: started thread_manager";
        // }}}

        // {{{ launch main
        // register the given main function with the thread manager
        lbt_ << "(1st stage) runtime::start: launching run_helper "
                "pika thread";

        threads::detail::thread_init_data data(
            util::detail::bind(
                &runtime::run_helper, this, func, std::ref(result_), true),
            "run_helper", execution::thread_priority::normal,
            execution::thread_schedule_hint(0),
            execution::thread_stacksize::large);

        this->runtime::starting();
        threads::detail::thread_id_ref_type id =
            threads::detail::invalid_thread_id;
        thread_manager_->register_thread(data, id);

        // }}}

        // block if required
        if (blocking)
        {
            return wait();    // wait for the shutdown_action to be executed
        }
        else
        {
            // wait for at least runtime_state::running
            util::yield_while(
                [this]() { return get_state() < runtime_state::running; },
                "runtime::start");
        }

        return 0;    // return zero as we don't know the outcome of pika_main yet
    }

    int runtime::start(bool blocking)
    {
        util::detail::function<pika_main_function_type> empty_main;
        return start(empty_main, blocking);
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::notify_finalize()
    {
        std::unique_lock<std::mutex> l(mtx_);
        if (!stop_called_)
        {
            stop_called_ = true;
            stop_done_ = true;
            wait_condition_.notify_all();
        }
    }

    void runtime::wait_finalize()
    {
        std::unique_lock<std::mutex> l(mtx_);
        while (!stop_done_)
        {
            LRT_(info).format("runtime: about to enter wait state");
            wait_condition_.wait(l);
            LRT_(info).format("runtime: exiting wait state");
        }
    }

    void runtime::wait_helper(
        std::mutex& mtx, std::condition_variable& cond, bool& running)
    {
        // signal successful initialization
        {
            std::lock_guard<std::mutex> lk(mtx);
            running = true;
            cond.notify_all();
        }

        // register this thread with any possibly active Intel tool
        std::string thread_name("main-thread#wait_helper");
        PIKA_ITT_THREAD_SET_NAME(thread_name.c_str());

        // set thread name as shown in Visual Studio
        detail::set_thread_name(thread_name.c_str());

#if defined(PIKA_HAVE_APEX)
        // not registering helper threads - for now
        //detail::external_timer::register_thread(thread_name.c_str());
#endif

        wait_finalize();
    }

    int runtime::wait()
    {
        LRT_(info).format("runtime: about to enter wait state");

        // start the wait_helper in a separate thread
        std::mutex mtx;
        std::condition_variable cond;
        bool running = false;

        std::thread t(util::detail::bind(&runtime::wait_helper, this,
            std::ref(mtx), std::ref(cond), std::ref(running)));

        // wait for the thread to run
        {
            std::unique_lock<std::mutex> lk(mtx);
            // NOLINTNEXTLINE(bugprone-infinite-loop)
            while (!running)    // -V776 // -V1044
                cond.wait(lk);
        }

        // block main thread
        t.join();

        thread_manager_->wait();

        LRT_(info).format("runtime: exiting wait state");
        return result_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // First half of termination process: stop thread manager,
    void runtime::stop(bool blocking)
    {
        LRT_(warning).format("runtime: about to stop services");

        call_shutdown_functions(true);

        // execute all on_exit functions whenever the first thread calls this
        this->runtime::stopping();

        // stop runtime services (threads)
        thread_manager_->stop(false);    // just initiate shutdown

#ifdef PIKA_HAVE_APEX
        detail::external_timer::finalize();
#endif

        if (threads::detail::get_self_ptr())
        {
            // schedule task on separate thread to execute stop_helper() below
            // this is necessary as this function (stop()) might have been called
            // from a pika thread, so it would deadlock by waiting for the thread
            // manager
            std::mutex mtx;
            std::condition_variable cond;
            std::unique_lock<std::mutex> l(mtx);

            std::thread t(util::detail::bind(&runtime::stop_helper, this,
                blocking, std::ref(cond), std::ref(mtx)));
            cond.wait(l);

            t.join();
        }
        else
        {
            thread_manager_->stop(blocking);    // wait for thread manager

            deinit_global_data();

            // this disables all logging from the main thread
            deinit_tss_helper("main-thread", 0);

            LRT_(info).format("runtime: stopped all services");
        }

        call_shutdown_functions(false);
    }

    // Second step in termination: shut down all services.
    void runtime::stop_helper(
        bool blocking, std::condition_variable& cond, std::mutex& mtx)
    {
        // wait for thread manager to exit
        thread_manager_->stop(blocking);    // wait for thread manager

        deinit_global_data();

        // this disables all logging from the main thread
        deinit_tss_helper("main-thread", 0);

        LRT_(info).format("runtime: stopped all services");

        std::lock_guard<std::mutex> l(mtx);
        cond.notify_all();    // we're done now
    }

    int runtime::suspend()
    {
        LRT_(info).format("runtime: about to suspend runtime");

        if (state_.load() == runtime_state::sleeping)
        {
            return 0;
        }

        if (state_.load() != runtime_state::running)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "runtime::suspend",
                "Can only suspend runtime from running state");
            return -1;
        }

        thread_manager_->suspend();

        set_state(runtime_state::sleeping);

        return 0;
    }

    int runtime::resume()
    {
        LRT_(info).format("runtime: about to resume runtime");

        if (state_.load() == runtime_state::running)
        {
            return 0;
        }

        if (state_.load() != runtime_state::sleeping)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "runtime::resume",
                "Can only resume runtime from suspended state");
            return -1;
        }

        thread_manager_->resume();

        set_state(runtime_state::running);

        return 0;
    }

    int runtime::finalize(double /*shutdown_timeout*/)
    {
        notify_finalize();
        return 0;
    }

    bool runtime::is_networking_enabled()
    {
        return false;
    }

    pika::threads::detail::thread_manager& runtime::get_thread_manager()
    {
        return *thread_manager_;
    }

    std::string runtime::here() const
    {
        return "127.0.0.1";
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime::report_error(std::size_t num_thread,
        std::exception_ptr const& e, bool /*terminate_all*/)
    {
        // call thread-specific user-supplied on_error handler
        bool report_exception = true;
        if (on_error_func_)
        {
            report_exception = on_error_func_(num_thread, e);
        }

        // Early and late exceptions, errors outside of pika-threads
        if (!threads::detail::get_self_ptr() ||
            !threads::thread_manager_is(runtime_state::running))
        {
            // report the error to the local console
            if (report_exception)
            {
                detail::report_exception_and_continue(e);
            }

            // store the exception to be able to rethrow it later
            {
                std::lock_guard<std::mutex> l(mtx_);
                exception_ = e;
            }

            notify_finalize();
            stop(false);

            return report_exception;
        }

        return report_exception;
    }

    bool runtime::report_error(std::exception_ptr const& e, bool terminate_all)
    {
        return report_error(pika::get_worker_thread_num(), e, terminate_all);
    }

    void runtime::rethrow_exception()
    {
        if (state_.load() > runtime_state::running)
        {
            std::lock_guard<std::mutex> l(mtx_);
            if (exception_)
            {
                std::exception_ptr e = exception_;
                exception_ = std::exception_ptr();
                std::rethrow_exception(e);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    int runtime::run(
        util::detail::function<pika_main_function_type> const& func)
    {
        // start the main thread function
        start(func);

        // now wait for everything to finish
        wait();
        stop();

        rethrow_exception();
        return result_;
    }

    ///////////////////////////////////////////////////////////////////////////
    int runtime::run()
    {
        // start the main thread function
        start();

        // now wait for everything to finish
        int result = wait();
        stop();

        rethrow_exception();
        return result;
    }

    util::thread_mapper& runtime::get_thread_mapper()
    {
        return *thread_support_;
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::callback_notifier runtime::get_notification_policy(
        char const* prefix, os_thread_type type)
    {
        using report_error_t =
            bool (runtime::*)(std::size_t, std::exception_ptr const&, bool);

        using std::placeholders::_1;
        using std::placeholders::_2;
        using std::placeholders::_3;
        using std::placeholders::_4;

        notification_policy_type notifier;

        notifier.add_on_start_thread_callback(
            util::detail::bind(&runtime::init_tss_helper, this, prefix, type,
                _1, _2, _3, _4, false));
        notifier.add_on_stop_thread_callback(
            util::detail::bind(&runtime::deinit_tss_helper, this, prefix, _1));
        notifier.set_on_error_callback(util::detail::bind(
            static_cast<report_error_t>(&runtime::report_error), this, _1, _2,
            true));

        return notifier;
    }

    void runtime::init_tss_helper(char const* context, os_thread_type type,
        std::size_t local_thread_num, std::size_t global_thread_num,
        char const* pool_name, char const* postfix, bool service_thread)
    {
        error_code ec(throwmode::lightweight);
        return init_tss_ex(context, type, local_thread_num, global_thread_num,
            pool_name, postfix, service_thread, ec);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void runtime::init_tss_ex(char const* context, os_thread_type type,
        std::size_t local_thread_num, std::size_t global_thread_num,
        char const* pool_name, char const* postfix, bool service_thread,
        error_code& ec)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        std::ostringstream fullname;
        fullname << "pika/" << context;
        if (pool_name && *pool_name)
        {
            fullname << "/pool:" << pool_name;
        }
        if (postfix && *postfix)
        {
            fullname << '/' << postfix;
        }
        if (global_thread_num != std::size_t(-1))
        {
            fullname << "/global:" + std::to_string(global_thread_num);
        }
        if (local_thread_num != std::size_t(-1))
        {
            fullname << "/local:" + std::to_string(local_thread_num);
        }

        PIKA_ASSERT(detail::thread_name().empty());
        detail::thread_name() = PIKA_MOVE(fullname).str();

        char const* name = detail::thread_name().c_str();

        // initialize thread mapping for external libraries (i.e. PAPI)
        thread_support_->register_thread(name, type);

        // register this thread with any possibly active Intel tool
        PIKA_ITT_THREAD_SET_NAME(name);

        // set thread name as shown in Visual Studio
        detail::set_thread_name(name);

#if defined(PIKA_HAVE_APEX)
        if (std::strstr(name, "worker") != nullptr)
            detail::external_timer::register_thread(name);
#endif

#ifdef PIKA_HAVE_TRACY
        tracy::SetThreadName(name);
#endif

        // call thread-specific user-supplied on_start handler
        if (on_start_func_)
        {
            on_start_func_(
                local_thread_num, global_thread_num, pool_name, context);
        }

        // if this is a service thread, set its service affinity
        if (service_thread)
        {
            // FIXME: We don't set the affinity of the service threads on BG/Q,
            // as this is causing a hang (needs to be investigated)
#if !defined(__bgq__)
            threads::detail::mask_cref_type used_processing_units =
                thread_manager_->get_used_processing_units();

            // --pika:bind=none  should disable all affinity definitions
            if (threads::detail::any(used_processing_units))
            {
                this->topology_.set_thread_affinity_mask(
                    this->topology_.get_service_affinity_mask(
                        used_processing_units),
                    ec);

                // comment this out for now as on CircleCI this is causing
                // unending grief
                // if (ec)
                // {
                //     PIKA_THROW_EXCEPTION(pika::error::kernel_error,
                //         "runtime::init_tss_ex",
                //         "failed to set thread affinity mask ({}) for service "
                //         "thread: {}",
                //         pika::threads::detail::to_string(used_processing_units),
                //         detail::thread_name());
                // }
            }
#endif
        }
    }

    void runtime::deinit_tss_helper(
        char const* context, std::size_t global_thread_num)
    {
        threads::detail::reset_continuation_recursion_count();

        // call thread-specific user-supplied on_stop handler
        if (on_stop_func_)
        {
            on_stop_func_(global_thread_num, global_thread_num, "", context);
        }

        // reset PAPI support
        thread_support_->unregister_thread();

        // reset thread local storage
        detail::thread_name().clear();
    }

    void runtime::add_pre_startup_function(startup_function_type f)
    {
        std::lock_guard<std::mutex> l(mtx_);
        pre_startup_functions_.push_back(PIKA_MOVE(f));
    }

    void runtime::add_startup_function(startup_function_type f)
    {
        std::lock_guard<std::mutex> l(mtx_);
        startup_functions_.push_back(PIKA_MOVE(f));
    }

    void runtime::add_pre_shutdown_function(shutdown_function_type f)
    {
        std::lock_guard<std::mutex> l(mtx_);
        pre_shutdown_functions_.push_back(PIKA_MOVE(f));
    }

    void runtime::add_shutdown_function(shutdown_function_type f)
    {
        std::lock_guard<std::mutex> l(mtx_);
        shutdown_functions_.push_back(PIKA_MOVE(f));
    }

    /// Register an external OS-thread with pika
    bool runtime::register_thread(char const* name,
        std::size_t global_thread_num, bool service_thread, error_code& ec)
    {
        std::string thread_name(name);
        thread_name += "-thread";

        init_tss_ex(thread_name.c_str(), os_thread_type::custom_thread,
            global_thread_num, global_thread_num, "", nullptr, service_thread,
            ec);

        return !ec ? true : false;
    }

    /// Unregister an external OS-thread with pika
    bool runtime::unregister_thread()
    {
        deinit_tss_helper(
            detail::thread_name().c_str(), pika::get_worker_thread_num());
        return true;
    }

    // Access data for a given OS thread that was previously registered by
    // \a register_thread. This function must be called from a thread that was
    // previously registered with the runtime.
    os_thread_data runtime::get_os_thread_data(std::string const& label) const
    {
        return thread_support_->get_os_thread_data(label);
    }

    /// Enumerate all OS threads that have registered with the runtime.
    bool runtime::enumerate_os_threads(
        util::detail::function<bool(os_thread_data const&)> const& f) const
    {
        return thread_support_->enumerate_os_threads(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::callback_notifier get_notification_policy(char const* prefix)
    {
        return get_runtime().get_notification_policy(
            prefix, os_thread_type::worker_thread);
    }

    std::uint32_t get_locality_id(error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt || rt->get_state() == runtime_state::invalid)
        {
            // same as naming::invalid_locality_id
            return ~static_cast<std::uint32_t>(0);
        }

        return rt->get_locality_id(ec);
    }

    std::size_t get_num_worker_threads()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "pika::get_num_worker_threads",
                "the runtime system has not been initialized yet");
            return std::size_t(0);
        }

        return rt->get_num_worker_threads();
    }

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    std::uint32_t get_num_localities(pika::launch::sync_policy, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "pika::get_num_localities",
                "the runtime system has not been initialized yet");
            return std::size_t(0);
        }

        return rt->get_num_localities(pika::launch::sync, ec);
    }

    std::uint32_t get_initial_num_localities()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "pika::get_initial_num_localities",
                "the runtime system has not been initialized yet");
            return std::size_t(0);
        }

        return rt->get_initial_num_localities();
    }

    pika::future<std::uint32_t> get_num_localities()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                "pika::get_num_localities",
                "the runtime system has not been initialized yet");
            return make_ready_future(std::uint32_t(0));
        }

        return rt->get_num_localities();
    }

    namespace threads {
        char const* get_stack_size_name(std::ptrdiff_t size)
        {
            execution::thread_stacksize size_enum =
                execution::thread_stacksize::unknown;

            pika::util::runtime_configuration const& rtcfg = pika::get_config();
            if (rtcfg.get_stack_size(execution::thread_stacksize::small_) ==
                size)
                size_enum = execution::thread_stacksize::small_;
            else if (rtcfg.get_stack_size(
                         execution::thread_stacksize::medium) == size)
                size_enum = execution::thread_stacksize::medium;
            else if (rtcfg.get_stack_size(execution::thread_stacksize::large) ==
                size)
                size_enum = execution::thread_stacksize::large;
            else if (rtcfg.get_stack_size(execution::thread_stacksize::huge) ==
                size)
                size_enum = execution::thread_stacksize::huge;
            else if (rtcfg.get_stack_size(
                         execution::thread_stacksize::nostack) == size)
                size_enum = execution::thread_stacksize::nostack;

            return execution::detail::get_stack_size_enum_name(size_enum);
        }
    }    // namespace threads
}    // namespace pika
