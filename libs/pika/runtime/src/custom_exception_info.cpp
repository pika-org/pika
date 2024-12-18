//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/modules/debugging.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/modules/threading.hpp>
#include <pika/runtime/config_entry.hpp>
#include <pika/runtime/custom_exception_info.hpp>
#include <pika/runtime/debugging.hpp>
#include <pika/runtime/get_worker_thread_num.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/state.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/version.hpp>

#if defined(PIKA_HAVE_MPI)
# include <pika/mpi_base/mpi.hpp>
# include <pika/mpi_base/mpi_environment.hpp>
#endif

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

#if defined(PIKA_WINDOWS)
# include <process.h>
#elif defined(PIKA_HAVE_UNISTD_H)
# include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace pika::detail {
    ///////////////////////////////////////////////////////////////////////////
    // For testing purposes we sometime expect to see exceptions, allow those
    // to go through without attaching a debugger.
    std::atomic<bool> expect_exception_flag(false);

    bool expect_exception(bool flag) { return expect_exception_flag.exchange(flag); }

    ///////////////////////////////////////////////////////////////////////////
    // Extract the diagnostic information embedded in the given exception and
    // return a string holding a formatted message.
    std::string diagnostic_information(pika::exception_info const& xi)
    {
        int const verbosity =
            detail::from_string<int>(get_config_entry("pika.exception_verbosity", "1"));

        std::ostringstream strm;
        strm << "\n";

        // add full build information
        if (verbosity >= 2)
        {
            strm << pika::detail::get_full_build_string();

            std::string const* env = xi.get<pika::detail::throw_env>();
            if (env && !env->empty()) strm << "{env}: " << *env;
        }

        if (verbosity >= 1)
        {
            std::string const* back_trace = xi.get<pika::detail::throw_stacktrace>();
            if (back_trace && !back_trace->empty())
            {
                // FIXME: add indentation to stack frame information
                strm << "{stack-trace}: " << *back_trace << "\n";
            }
#if defined(PIKA_HAVE_MPI)
            if (mpi::detail::environment::is_mpi_initialized())
                strm << "MPI rank: " << mpi::detail::environment::rank() << '\n';
#endif
            std::string const* hostname_ = xi.get<pika::detail::throw_hostname>();
            if (hostname_ && !hostname_->empty()) strm << "{hostname}: " << *hostname_ << "\n";

            std::int64_t const* pid_ = xi.get<pika::detail::throw_pid>();
            if (pid_ && -1 != *pid_) strm << "{process-id}: " << *pid_ << "\n";

            bool thread_info = false;
            char const* const thread_prefix = "{os-thread}: ";
            std::size_t const* shepherd = xi.get<pika::detail::throw_shepherd>();
            if (shepherd && std::size_t(-1) != *shepherd)
            {
                strm << thread_prefix << *shepherd;
                thread_info = true;
            }

            std::string thread_name = pika::detail::get_thread_name();
            if (!thread_info)
                strm << thread_prefix;
            else
                strm << ", ";
            strm << thread_name << "\n";

            std::size_t const* thread_id = xi.get<pika::detail::throw_thread_id>();
            if (thread_id && *thread_id)
            {
                strm << "{thread-id}: ";
                fmt::print(strm, "{:016x}\n", *thread_id);
            }

            std::string const* thread_description = xi.get<pika::detail::throw_thread_name>();
            if (thread_description && !thread_description->empty())
                strm << "{thread-description}: " << *thread_description << "\n";

            std::string const* state = xi.get<pika::detail::throw_state>();
            if (state) strm << "{state}: " << *state << "\n";

            std::string const* auxinfo = xi.get<pika::detail::throw_auxinfo>();
            if (auxinfo) strm << "{auxinfo}: " << *auxinfo << "\n";
        }

        std::string const* file = xi.get<pika::detail::throw_file>();
        if (file) strm << "{file}: " << *file << "\n";

        long const* line = xi.get<pika::detail::throw_line>();
        if (line) strm << "{line}: " << *line << "\n";

        std::string const* function = xi.get<pika::detail::throw_function>();
        if (function) strm << "{function}: " << *function << "\n";

        // Try a cast to std::exception - this should handle boost.system
        // error codes in addition to the standard library exceptions.
        std::exception const* se = dynamic_cast<std::exception const*>(&xi);
        if (se) strm << "{what}: " << se->what() << "\n";

        return strm.str();
    }

    void pre_exception_handler()
    {
        if (!expect_exception_flag.load(std::memory_order_relaxed))
        {
            pika::detail::may_attach_debugger("exception");
        }
    }

    static get_full_build_string_type get_full_build_string_f;

    void set_get_full_build_string(get_full_build_string_type f) { get_full_build_string_f = f; }

    std::string get_full_build_string()
    {
        if (detail::get_full_build_string_f) { return detail::get_full_build_string_f(); }
        else { return pika::full_build_string(); }
    }

    ///////////////////////////////////////////////////////////////////////////
    // report an early or late exception and abort
    void report_exception_and_continue(std::exception const& e)
    {
        pre_exception_handler();

        std::cerr << e.what() << std::endl;
    }

    void report_exception_and_continue(std::exception_ptr const& e)
    {
        pre_exception_handler();

        std::cerr << diagnostic_information(e) << std::endl;
    }

    void report_exception_and_continue(pika::exception const& e)
    {
        pre_exception_handler();

        std::cerr << diagnostic_information(e) << std::endl;
    }

    void report_exception_and_terminate(std::exception const& e)
    {
        report_exception_and_continue(e);
        std::abort();
    }

    void report_exception_and_terminate(std::exception_ptr const& e)
    {
        report_exception_and_continue(e);
        std::abort();
    }

    void report_exception_and_terminate(pika::exception const& e)
    {
        report_exception_and_continue(e);
        std::abort();
    }

    pika::exception_info construct_exception_info(std::string const& func, std::string const& file,
        long line, std::string const& back_trace, std::string const& hostname, std::int64_t pid,
        std::size_t shepherd, std::size_t thread_id, std::string const& thread_name,
        std::string const& env, std::string const& config, std::string const& state_name,
        std::string const& auxinfo)
    {
        return pika::exception_info().set(pika::detail::throw_stacktrace(back_trace),
            pika::detail::throw_hostname(hostname), pika::detail::throw_pid(pid),
            pika::detail::throw_shepherd(shepherd), pika::detail::throw_thread_id(thread_id),
            pika::detail::throw_thread_name(thread_name), pika::detail::throw_function(func),
            pika::detail::throw_file(file), pika::detail::throw_line(line),
            pika::detail::throw_env(env), pika::detail::throw_config(config),
            pika::detail::throw_state(state_name), pika::detail::throw_auxinfo(auxinfo));
    }

    template <typename Exception>
    std::exception_ptr construct_exception(Exception const& e, pika::exception_info info)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try
        {
            throw_with_info(e, std::move(info));
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        PIKA_ASSERT(false);    // -V779
        return std::exception_ptr();
    }

    template PIKA_EXPORT std::exception_ptr construct_exception(
        pika::exception const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::system_error const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::exception const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        pika::detail::std_exception const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::bad_exception const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        pika::detail::bad_exception const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::bad_typeid const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        pika::detail::bad_typeid const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::bad_cast const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        pika::detail::bad_cast const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::bad_alloc const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        pika::detail::bad_alloc const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::logic_error const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::runtime_error const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::out_of_range const&, pika::exception_info info);
    template PIKA_EXPORT std::exception_ptr construct_exception(
        std::invalid_argument const&, pika::exception_info info);

    ///////////////////////////////////////////////////////////////////////////
    //  Figure out the size of the given environment
    inline std::size_t get_arraylen(char** array)
    {
        std::size_t count = 0;
        if (nullptr != array)
        {
            while (nullptr != array[count]) ++count;    // simply count the environment strings
        }
        return count;
    }

    std::string get_execution_environment()
    {
        std::vector<std::string> env;

#if defined(PIKA_WINDOWS)
        std::size_t len = get_arraylen(_environ);
        env.reserve(len);
        std::copy(&_environ[0], &_environ[len], std::back_inserter(env));
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__AIX__)
        std::size_t len = get_arraylen(environ);
        env.reserve(len);
        std::copy(&environ[0], &environ[len], std::back_inserter(env));
#elif defined(__FreeBSD__)
        std::size_t len = get_arraylen(freebsd_environ);
        env.reserve(len);
        std::copy(&freebsd_environ[0], &freebsd_environ[len], std::back_inserter(env));
#elif defined(__APPLE__)
        std::size_t len = get_arraylen(environ);
        env.reserve(len);
        std::copy(&environ[0], &environ[len], std::back_inserter(env));
#else
# error "Don't know, how to access the execution environment on this platform"
#endif

        std::sort(env.begin(), env.end());

        static constexpr char const* ignored_env_patterns[] = {"DOCKER", "GITHUB_TOKEN"};
        std::string retval = fmt::format("{} entries:\n", env.size());
        for (std::string const& s : env)
        {
            if (std::all_of(std::begin(ignored_env_patterns), std::end(ignored_env_patterns),
                    [&s](auto const e) { return s.find(e) == std::string::npos; }))
            {
                retval += "  " + s + "\n";
            }
        }
        return retval;
    }

    pika::exception_info custom_exception_info(
        std::string const& func, std::string const& file, long line, std::string const& auxinfo)
    {
        std::int64_t pid = ::getpid();

        std::size_t const trace_depth = detail::from_string<std::size_t>(
            get_config_entry("pika.trace_depth", PIKA_HAVE_THREAD_BACKTRACE_DEPTH));

        std::string back_trace = pika::debug::detail::trace(trace_depth);

        std::string state_name("not running");
        std::string hostname;
        runtime* rt = get_runtime_ptr();
        if (rt)
        {
            pika::runtime_state rts_state = rt->get_state();
            state_name = pika::detail::get_runtime_state_name(rts_state);

            if (rts_state >= pika::runtime_state::initialized &&
                rts_state < pika::runtime_state::stopped)
            {
                hostname = pika::debug::detail::hostname_print_helper{}.get_hostname_and_rank();
            }
        }

        // if this is not a pika thread we do not need to query neither for
        // the shepherd thread nor for the thread id
        error_code ec(throwmode::lightweight);

        std::size_t shepherd = std::size_t(-1);
        pika::threads::detail::thread_id_type thread_id;
        detail::thread_description thread_name;

        pika::threads::detail::thread_self* self = pika::threads::detail::get_self_ptr();
        if (nullptr != self)
        {
            if (thread_manager_is(pika::runtime_state::running))
                shepherd = pika::get_worker_thread_num();

            thread_id = pika::threads::detail::get_self_id();
            thread_name = pika::threads::detail::get_thread_description(thread_id);
        }

        std::string env(pika::detail::get_execution_environment());
        std::string config(pika::configuration_string());

        return pika::exception_info().set(pika::detail::throw_stacktrace(back_trace),
            pika::detail::throw_hostname(hostname), pika::detail::throw_pid(pid),
            pika::detail::throw_shepherd(shepherd),
            pika::detail::throw_thread_id(reinterpret_cast<std::size_t>(thread_id.get())),
            pika::detail::throw_thread_name(detail::as_string(thread_name)),
            pika::detail::throw_function(func), pika::detail::throw_file(file),
            pika::detail::throw_line(line), pika::detail::throw_env(env),
            pika::detail::throw_config(config), pika::detail::throw_state(state_name),
            pika::detail::throw_auxinfo(auxinfo));
    }

    /// Return the host-name of the process where the exception was thrown.
    std::string get_error_host_name(pika::exception_info const& xi)
    {
        std::string const* hostname_ = xi.get<pika::detail::throw_hostname>();
        if (hostname_ && !hostname_->empty()) return *hostname_;
        return std::string();
    }

    /// Return the (operating system) process id of the process where the
    /// exception was thrown.
    std::int64_t get_error_process_id(pika::exception_info const& xi)
    {
        std::int64_t const* pid_ = xi.get<pika::detail::throw_pid>();
        if (pid_) return *pid_;
        return -1;
    }

    /// Return the environment of the OS-process at the point the exception
    /// was thrown.
    std::string get_error_env(pika::exception_info const& xi)
    {
        std::string const* env = xi.get<pika::detail::throw_env>();
        if (env && !env->empty()) return *env;

        return "<unknown>";
    }

    /// Return the stack backtrace at the point the exception was thrown.
    std::string get_error_backtrace(pika::exception_info const& xi)
    {
        std::string const* back_trace = xi.get<pika::detail::throw_stacktrace>();
        if (back_trace && !back_trace->empty()) return *back_trace;

        return std::string();
    }

    /// Return the sequence number of the OS-thread used to execute pika-threads
    /// from which the exception was thrown.
    std::size_t get_error_os_thread(pika::exception_info const& xi)
    {
        std::size_t const* shepherd = xi.get<pika::detail::throw_shepherd>();
        if (shepherd && std::size_t(-1) != *shepherd) return *shepherd;
        return std::size_t(-1);
    }

    /// Return the unique thread id of the pika-thread from which the exception
    /// was thrown.
    std::size_t get_error_thread_id(pika::exception_info const& xi)
    {
        std::size_t const* thread_id = xi.get<pika::detail::throw_thread_id>();
        if (thread_id && *thread_id) return *thread_id;
        return std::size_t(-1);
    }

    /// Return any addition thread description of the pika-thread from which the
    /// exception was thrown.
    std::string get_error_thread_description(pika::exception_info const& xi)
    {
        std::string const* thread_description = xi.get<pika::detail::throw_thread_name>();
        if (thread_description && !thread_description->empty()) return *thread_description;
        return std::string();
    }

    /// Return the pika configuration information point from which the
    /// exception was thrown.
    std::string get_error_config(pika::exception_info const& xi)
    {
        std::string const* config_info = xi.get<pika::detail::throw_config>();
        if (config_info && !config_info->empty()) return *config_info;
        return std::string();
    }

    /// Return the pika runtime state information at which the exception was
    /// thrown.
    std::string get_error_state(pika::exception_info const& xi)
    {
        std::string const* state_info = xi.get<pika::detail::throw_state>();
        if (state_info && !state_info->empty()) return *state_info;
        return std::string();
    }
}    // namespace pika::detail
