//  Copyright (c) 2005-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/detail/filesystem.hpp>
#include <pika/errors/exception.hpp>
#include <pika/preprocessor/expand.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/runtime_configuration/init_ini_data.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/string_util/classification.hpp>
#include <pika/string_util/from_string.hpp>
#include <pika/string_util/split.hpp>
#include <pika/type_support/unused.hpp>
#include <pika/util/get_entry_as.hpp>
#include <pika/version.hpp>

#include <boost/tokenizer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(PIKA_WINDOWS)
# include <process.h>
#elif defined(PIKA_HAVE_UNISTD_H)
# include <unistd.h>
#endif

#include <limits>

#if defined(PIKA_WINDOWS)
# include <windows.h>
#elif defined(__linux) || defined(linux) || defined(__linux__)
# include <linux/limits.h>
# include <sys/stat.h>
# include <unistd.h>
# include <vector>
#elif __APPLE__
# include <mach-o/dyld.h>
#elif defined(__FreeBSD__)
# include <algorithm>
# include <iterator>
# include <sys/sysctl.h>
# include <sys/types.h>
# include <vector>
#endif

namespace pika::detail {

    std::string get_executable_filename(char const* argv0)
    {
        std::string r;

#if defined(PIKA_WINDOWS)
        PIKA_UNUSED(argv0);

        char exe_path[MAX_PATH + 1] = {'\0'};
        if (!GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path)))
        {
            PIKA_THROW_EXCEPTION(pika::error::dynamic_link_failure, "get_executable_filename",
                "unable to find executable filename");
        }
        r = exe_path;

#elif defined(__linux) || defined(linux) || defined(__linux__)
        char buf[PATH_MAX + 1];
        ssize_t length = ::readlink("/proc/self/exe", buf, sizeof(buf));

        if (length != -1)
        {
            buf[length] = '\0';
            r = buf;
            return r;
        }

        std::string argv0_(argv0);

        // REVIEW: Should we resolve symlinks at any point here?
        if (argv0_.length() > 0)
        {
            // Check for an absolute path.
            if (argv0_[0] == '/') return argv0_;

            // Check for a relative path.
            if (argv0_.find('/') != std::string::npos)
            {
                // Get the current working directory.

                // NOTE: getcwd does give you a null terminated string,
                // while readlink (above) does not.
                if (::getcwd(buf, PATH_MAX))
                {
                    r = buf;
                    r += '/';
                    r += argv0_;
                    return r;
                }
            }

            // Search PATH
            char const* epath = ::getenv("PATH");
            if (epath)
            {
                std::vector<std::string> path_dirs;

                pika::detail::split(path_dirs, epath, pika::detail::is_any_of(":"),
                    pika::detail::token_compress_mode::on);

                for (std::uint64_t i = 0; i < path_dirs.size(); ++i)
                {
                    r = path_dirs[i];
                    r += '/';
                    r += argv0_;

                    // Can't use Boost.Filesystem as it doesn't let me access
                    // st_uid and st_gid.
                    struct stat s;

                    // Make sure the file is executable and shares our
                    // effective uid and gid.
                    // NOTE: If someone was using a pika application that was
                    // seteuid'd to root, this may fail.
                    if (0 == ::stat(r.c_str(), &s))
                        if ((s.st_uid == ::geteuid()) && (s.st_mode & S_IXUSR) &&
                            (s.st_gid == ::getegid()) && (s.st_mode & S_IXGRP) &&
                            (s.st_mode & S_IXOTH))
                            return r;
                }
            }
        }

        PIKA_THROW_EXCEPTION(pika::error::dynamic_link_failure, "get_executable_filename",
            "unable to find executable filename");

#elif defined(__APPLE__)
        PIKA_UNUSED(argv0);

        char exe_path[PATH_MAX + 1];
        std::uint32_t len = sizeof(exe_path) / sizeof(exe_path[0]);

        if (0 != _NSGetExecutablePath(exe_path, &len))
        {
            PIKA_THROW_EXCEPTION(pika::error::dynamic_link_failure, "get_executable_filename",
                "unable to find executable filename");
        }
        exe_path[len - 1] = '\0';
        r = exe_path;

#elif defined(__FreeBSD__)
        PIKA_UNUSED(argv0);

        int mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
        size_t cb = 0;
        sysctl(mib, 4, nullptr, &cb, nullptr, 0);
        if (cb)
        {
            std::vector<char> buf(cb);
            sysctl(mib, 4, &buf[0], &cb, nullptr, 0);
            std::copy(buf.begin(), buf.end(), std::back_inserter(r));
        }

#else
# error Unsupported platform
#endif

        return r;
    }

    std::string get_executable_prefix(char const* argv0 = nullptr)
    {
        std::filesystem::path p(get_executable_filename(argv0));

        return p.parent_path().parent_path().string();
    }

}    // namespace pika::detail

namespace pika::util {

    namespace detail {

        // CMake does not deal with explicit semicolons well, for this reason,
        // the paths are delimited with ':'. On Windows those need to be
        // converted to ';'.
        std::string convert_delimiters(std::string paths)
        {
#if defined(PIKA_WINDOWS)
            std::replace(paths.begin(), paths.end(), ':', ';');
#endif
            return paths;
        }
    }    // namespace detail

    // pre-initialize entries with compile time based values
    void runtime_configuration::pre_initialize_ini()
    {
        if (!need_to_call_pre_initialize) return;

        std::vector<std::string> lines = {
            // clang-format off
            // create an empty application section
            "[application]",

            // create system and application instance specific entries
            "[system]",
            "pid = " + std::to_string(getpid()),
#if defined(__linux) || defined(linux) || defined(__linux__)
            "executable_prefix = " + pika::detail::get_executable_prefix(argv0),
#else
            "executable_prefix = " + pika::detail::get_executable_prefix(),
#endif
            // create default installation location and logging settings
            "[pika]",
            "master_ini_path = $[system.executable_prefix]/",
            // NOLINTNEXTLINE(bugprone-suspicious-missing-comma)
            "master_ini_path_suffixes = /share/pika" PIKA_INI_PATH_DELIMITER
                "/../share/pika",
            "shutdown_check_count = ${PIKA_SHUTDOWN_CHECK_COUNT:10}",
#ifdef PIKA_HAVE_VERIFY_LOCKS
#if defined(PIKA_DEBUG)
            "lock_detection = ${PIKA_LOCK_DETECTION:1}",
#else
            "lock_detection = ${PIKA_LOCK_DETECTION:0}",
#endif
            "throw_on_held_lock = ${PIKA_THROW_ON_HELD_LOCK:1}",
#endif
#ifdef PIKA_HAVE_THREAD_DEADLOCK_DETECTION
#ifdef PIKA_DEBUG
            "deadlock_detection = ${PIKA_DEADLOCK_DETECTION:1}",
#else
            "deadlock_detection = ${PIKA_DEADLOCK_DETECTION:0}",
#endif
#endif
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
#ifdef PIKA_DEBUG
            "spinlock_deadlock_detection = "
            "${PIKA_SPINLOCK_DEADLOCK_DETECTION:1}",
#else
            "spinlock_deadlock_detection = "
            "${PIKA_SPINLOCK_DEADLOCK_DETECTION:0}",
#endif
            "spinlock_deadlock_detection_limit = "
            "${PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT)) "}",
            "spinlock_deadlock_warning_limit = "
            "${PIKA_SPINLOCK_DEADLOCK_WARNING_LIMIT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_SPINLOCK_DEADLOCK_WARNING_LIMIT)) "}",
#endif

            // add placeholders for keys to be added by command line handling
            "ignore_process_mask = 0",
            "process_mask = ${PIKA_PROCESS_MASK:}",
            "os_threads = cores",
            "cores = all",
            "first_pu = 0",
            "scheduler = local-priority-fifo",
            "affinity = core",
            "pu_step = 1",
            "pu_offset = 0",
            "numa_sensitive = 0",
            "max_idle_loop_count = ${PIKA_MAX_IDLE_LOOP_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_IDLE_LOOP_COUNT_MAX)) "}",
            "max_busy_loop_count = ${PIKA_MAX_BUSY_LOOP_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_BUSY_LOOP_COUNT_MAX)) "}",
#if defined(PIKA_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
            "max_idle_backoff_time = "
            "${PIKA_MAX_IDLE_BACKOFF_TIME:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_IDLE_BACKOFF_TIME_MAX)) "}",
#endif
            "default_scheduler_mode = ${PIKA_DEFAULT_SCHEDULER_MODE}",

            "install_signal_handlers = ${PIKA_INSTALL_SIGNAL_HANDLERS:0}",
            "diagnostics_on_terminate = ${PIKA_DIAGNOSTICS_ON_TERMINATE:1}",
            "attach_debugger = ${PIKA_ATTACH_DEBUGGER:0}",
            "exception_verbosity = ${PIKA_EXCEPTION_VERBOSITY:1}",
            "trace_depth = ${PIKA_TRACE_DEPTH:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_HAVE_THREAD_BACKTRACE_DEPTH)) "}",

            "[pika.stacks]",
            "small_size = ${PIKA_SMALL_STACK_SIZE:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_SMALL_STACK_SIZE)) "}",
            "medium_size = ${PIKA_MEDIUM_STACK_SIZE:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_MEDIUM_STACK_SIZE)) "}",
            "large_size = ${PIKA_LARGE_STACK_SIZE:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_LARGE_STACK_SIZE)) "}",
            "huge_size = ${PIKA_HUGE_STACK_SIZE:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_HUGE_STACK_SIZE)) "}",
#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__)
            "use_guard_pages = ${PIKA_USE_GUARD_PAGES:0}",
#endif

            "[pika.thread_queue]",
            "max_thread_count = ${PIKA_THREAD_QUEUE_MAX_THREAD_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MAX_THREAD_COUNT)) "}",
            "min_tasks_to_steal_pending = "
            "${PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING)) "}",
            "min_tasks_to_steal_staged = "
            "${PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED)) "}",
            "min_add_new_count = "
            "${PIKA_THREAD_QUEUE_MIN_ADD_NEW_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MIN_ADD_NEW_COUNT)) "}",
            "max_add_new_count = "
            "${PIKA_THREAD_QUEUE_MAX_ADD_NEW_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MAX_ADD_NEW_COUNT)) "}",
            "min_delete_count = "
            "${PIKA_THREAD_QUEUE_MIN_DELETE_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MIN_DELETE_COUNT)) "}",
            "max_delete_count = "
            "${PIKA_THREAD_QUEUE_MAX_DELETE_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MAX_THREAD_COUNT)) "}",
            "max_terminated_threads = "
            "${PIKA_THREAD_QUEUE_MAX_TERMINATED_THREADS:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_MAX_TERMINATED_THREADS)) "}",
            "init_threads_count = "
            "${PIKA_THREAD_QUEUE_INIT_THREADS_COUNT:" PIKA_PP_STRINGIZE(
                PIKA_PP_EXPAND(PIKA_THREAD_QUEUE_INIT_THREADS_COUNT)) "}",

#if defined(PIKA_HAVE_MPI)
            "[pika.mpi]",
            "enable_pool = ${PIKA_MPI_ENABLE_POOL:0}",
            "completion_mode = ${PIKA_MPI_COMPLETION_MODE:30}",
#endif

            "[pika.commandline]",

            // allow for unknown options to be passed through
            "allow_unknown = ${PIKA_COMMANDLINE_ALLOW_UNKNOWN:0}",

            // allow for command line options to to be passed through the
            // environment
            "prepend_options = ${PIKA_COMMANDLINE_OPTIONS}",
            // clang-format on
        };

        lines.insert(lines.end(), extra_static_ini_defs.begin(), extra_static_ini_defs.end());

        // don't overload user overrides
        this->parse("<static defaults>", lines, false, false, false);

        need_to_call_pre_initialize = false;
    }

    void runtime_configuration::post_initialize_ini(
        std::string& pika_ini_file_, std::vector<std::string> const& cmdline_ini_defs_)
    {
        util::init_ini_data_base(*this, pika_ini_file_);
        need_to_call_pre_initialize = true;

        // let the command line override the config file.
        if (!cmdline_ini_defs_.empty())
        {
            // do not weed out comments
            this->parse("<command line definitions>", cmdline_ini_defs_, true, false);
            need_to_call_pre_initialize = true;
        }
    }

    void runtime_configuration::pre_initialize_logging_ini()
    {
        std::vector<std::string> lines = {
            "[pika.log]",
            "level = ${PIKA_LOG_LEVEL:3}",
            "destination = ${PIKA_LOG_DESTINATION:cerr}",
            "format = ${PIKA_LOG_FORMAT:"
            "[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] [host:%j] [pid:%P] [tid:%t] "
            "[pool:%w] [parent:%q] [task:%k] [%s:%#/%!] %v"
            "}",
        };

        // don't overload user overrides
        this->parse("<static logging defaults>", lines, false, false);
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    runtime_configuration::runtime_configuration(
        char const* argv0_, std::vector<std::string> const& extra_static_ini_defs_)
      : extra_static_ini_defs(extra_static_ini_defs_)
      , num_os_threads(0)
      , small_stacksize(PIKA_SMALL_STACK_SIZE)
      , medium_stacksize(PIKA_MEDIUM_STACK_SIZE)
      , large_stacksize(PIKA_LARGE_STACK_SIZE)
      , huge_stacksize(PIKA_HUGE_STACK_SIZE)
      , need_to_call_pre_initialize(true)
#if defined(__linux) || defined(linux) || defined(__linux__)
      , argv0(argv0_)
#endif
    {
#if !(defined(__linux) || defined(linux) || defined(__linux__))
        PIKA_UNUSED(argv0_);
#endif
        pre_initialize_ini();

        // set global config options
        PIKA_ASSERT(init_small_stack_size() >= PIKA_SMALL_STACK_SIZE);

        small_stacksize = init_small_stack_size();
        medium_stacksize = init_medium_stack_size();
        large_stacksize = init_large_stack_size();
        PIKA_ASSERT(init_huge_stack_size() <= PIKA_HUGE_STACK_SIZE);
        huge_stacksize = init_huge_stack_size();
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_configuration::reconfigure(std::string const& pika_ini_file_)
    {
        pika_ini_file = pika_ini_file_;
        reconfigure();
    }

    void runtime_configuration::reconfigure(std::vector<std::string> const& cmdline_ini_defs_)
    {
        cmdline_ini_defs = cmdline_ini_defs_;
        reconfigure();
    }

    void runtime_configuration::reconfigure()
    {
        pre_initialize_ini();
        pre_initialize_logging_ini();
        post_initialize_ini(pika_ini_file, cmdline_ini_defs);

        // set global config options
        PIKA_ASSERT(init_small_stack_size() >= PIKA_SMALL_STACK_SIZE);

        small_stacksize = init_small_stack_size();
        medium_stacksize = init_medium_stack_size();
        large_stacksize = init_large_stack_size();
        huge_stacksize = init_huge_stack_size();
    }

    // Enable lock detection during suspension
    bool runtime_configuration::enable_lock_detection() const
    {
#ifdef PIKA_HAVE_VERIFY_LOCKS
        if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::detail::get_entry_as<int>(*sec, "lock_detection", 0) != 0;
        }
#endif
        return false;
    }

    // Enable minimal deadlock detection for pika threads
    bool runtime_configuration::enable_deadlock_detection() const
    {
#ifdef PIKA_HAVE_THREAD_DEADLOCK_DETECTION
        if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
        {
# ifdef PIKA_DEBUG
            return pika::detail::get_entry_as<int>(*sec, "deadlock_detection", 1) != 0;
# else
            return pika::detail::get_entry_as<int>(*sec, "deadlock_detection", 0) != 0;
# endif
        }

# ifdef PIKA_DEBUG
        return true;
# else
        return false;
# endif

#else
        return false;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_configuration::enable_spinlock_deadlock_detection() const
    {
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
        if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
        {
# ifdef PIKA_DEBUG
            return pika::detail::get_entry_as<int>(*sec, "spinlock_deadlock_detection", 1) != 0;
# else
            return pika::detail::get_entry_as<int>(*sec, "spinlock_deadlock_detection", 0) != 0;
# endif
        }

# ifdef PIKA_DEBUG
        return true;
# else
        return false;
# endif

#else
        return false;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t runtime_configuration::get_spinlock_deadlock_detection_limit() const
    {
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
        if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::detail::get_entry_as<std::size_t>(
                *sec, "spinlock_deadlock_detection_limit", PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT);
        }
        return PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT;
#else
        return std::size_t(-1);
#endif
    }

    std::size_t runtime_configuration::get_spinlock_deadlock_warning_limit() const
    {
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
        if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::detail::get_entry_as<std::size_t>(
                *sec, "spinlock_deadlock_warning_limit", PIKA_SPINLOCK_DEADLOCK_WARNING_LIMIT);
        }
        return PIKA_SPINLOCK_DEADLOCK_WARNING_LIMIT;
#else
        return std::size_t(-1);
#endif
    }

    std::size_t runtime_configuration::trace_depth() const
    {
        if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::detail::get_entry_as<std::size_t>(
                *sec, "trace_depth", PIKA_HAVE_THREAD_BACKTRACE_DEPTH);
        }
        return PIKA_HAVE_THREAD_BACKTRACE_DEPTH;
    }

    std::size_t runtime_configuration::get_os_thread_count() const
    {
        if (num_os_threads == 0)
        {
            num_os_threads = 1;
            if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
            {
                num_os_threads = pika::detail::get_entry_as<std::uint32_t>(*sec, "os_threads", 1);
            }
        }
        return static_cast<std::size_t>(num_os_threads);
    }

    std::string runtime_configuration::get_cmd_line() const
    {
        if (pika::detail::section const* sec = get_section("pika"); nullptr != sec)
        {
            return sec->get_entry("cmd_line", "");
        }
        return "";
    }

    // Return the configured sizes of any of the know thread pools
    std::size_t runtime_configuration::get_thread_pool_size(char const* poolname) const
    {
        if (pika::detail::section const* sec = get_section("pika.threadpools"); nullptr != sec)
        {
            return pika::detail::get_entry_as<std::size_t>(
                *sec, std::string(poolname) + "_size", 2);
        }
        return 2;    // the default size for all pools is 2
    }

    // Will return the stack size to use for all pika-threads.
    std::ptrdiff_t runtime_configuration::init_stack_size(
        char const* entryname, char const* defaultvaluestr, std::ptrdiff_t defaultvalue) const
    {
        if (pika::detail::section const* sec = get_section("pika.stacks"); nullptr != sec)
        {
            std::string entry = sec->get_entry(entryname, defaultvaluestr);
            char* endptr = nullptr;
            std::ptrdiff_t val = std::strtoll(entry.c_str(), &endptr, /*base:*/ 0);
            return endptr != entry.c_str() ? val : defaultvalue;
        }
        return defaultvalue;
    }

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
    bool runtime_configuration::use_stack_guard_pages() const
    {
        if (pika::detail::section const* sec = get_section("pika.stacks"); nullptr != sec)
        {
            return pika::detail::get_entry_as<int>(*sec, "use_guard_pages", 1) != 0;
        }
        return true;    // default is true
    }
#endif

    std::ptrdiff_t runtime_configuration::init_small_stack_size() const
    {
        return init_stack_size(
            "small_size", PIKA_PP_STRINGIZE(PIKA_SMALL_STACK_SIZE), PIKA_SMALL_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_medium_stack_size() const
    {
        return init_stack_size(
            "medium_size", PIKA_PP_STRINGIZE(PIKA_MEDIUM_STACK_SIZE), PIKA_MEDIUM_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_large_stack_size() const
    {
        return init_stack_size(
            "large_size", PIKA_PP_STRINGIZE(PIKA_LARGE_STACK_SIZE), PIKA_LARGE_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_huge_stack_size() const
    {
        return init_stack_size(
            "huge_size", PIKA_PP_STRINGIZE(PIKA_HUGE_STACK_SIZE), PIKA_HUGE_STACK_SIZE);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_configuration::load_application_configuration(char const* filename, error_code& ec)
    {
        try
        {
            section appcfg(filename);
            section applroot;
            applroot.add_section("application", appcfg);
            this->section::merge(applroot);
        }
        catch (pika::exception const& e)
        {
            // file doesn't exist or is ill-formed
            if (&ec == &throws) throw;
            ec = make_error_code(e.get_error(), e.what(), pika::throwmode::rethrow);
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::ptrdiff_t runtime_configuration::get_stack_size(
        execution::thread_stacksize stacksize) const
    {
        switch (stacksize)
        {
        case execution::thread_stacksize::medium: return medium_stacksize;

        case execution::thread_stacksize::large: return large_stacksize;

        case execution::thread_stacksize::huge: return huge_stacksize;

        case execution::thread_stacksize::nostack:
            return (std::numeric_limits<std::ptrdiff_t>::max)();

        default:
        case execution::thread_stacksize::small_: break;
        }
        return small_stacksize;
    }
}    // namespace pika::util
