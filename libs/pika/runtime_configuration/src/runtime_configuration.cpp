//  Copyright (c) 2005-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config/endian.hpp>
#include <pika/assert.hpp>
#include <pika/local/version.hpp>
#include <pika/modules/filesystem.hpp>
#include <pika/modules/itt_notify.hpp>
#include <pika/prefix/find_prefix.hpp>
#include <pika/preprocessor/expand.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/runtime_configuration/init_ini_data.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/runtime_configuration/runtime_mode.hpp>
#include <pika/util/from_string.hpp>
#include <pika/util/get_entry_as.hpp>

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
#include <process.h>
#elif defined(PIKA_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include <limits>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {

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
        if (!need_to_call_pre_initialize)
            return;

        std::vector<std::string> lines = {
            // clang-format off
            // create an empty application section
            "[application]",

            // create system and application instance specific entries
            "[system]",
            "pid = " + std::to_string(getpid()),
#if defined(__linux) || defined(linux) || defined(__linux__)
            "executable_prefix = " + get_executable_prefix(argv0),
#else
            "executable_prefix = " + get_executable_prefix(),
#endif
            // create default installation location and logging settings
            "[pika]",
            "master_ini_path = $[system.executable_prefix]/",
            "master_ini_path_suffixes = /share/pika" PIKA_INI_PATH_DELIMITER
                "/../share/pika",
#ifdef PIKA_HAVE_ITTNOTIFY
            "use_itt_notify = ${PIKA_HAVE_ITTNOTIFY:0}",
#endif
            "finalize_wait_time = ${PIKA_FINALIZE_WAIT_TIME:-1.0}",
            "shutdown_timeout = ${PIKA_SHUTDOWN_TIMEOUT:-1.0}",
            "shutdown_check_count = ${PIKA_SHUTDOWN_CHECK_COUNT:10}",
#ifdef PIKA_HAVE_VERIFY_LOCKS
#if defined(PIKA_DEBUG)
            "lock_detection = ${PIKA_LOCK_DETECTION:1}",
#else
            "lock_detection = ${PIKA_LOCK_DETECTION:0}",
#endif
            "throw_on_held_lock = ${PIKA_THROW_ON_HELD_LOCK:1}",
#endif
#ifdef PIKA_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
#ifdef PIKA_DEBUG
            "minimal_deadlock_detection = ${PIKA_MINIMAL_DEADLOCK_DETECTION:1}",
#else
            "minimal_deadlock_detection = ${PIKA_MINIMAL_DEADLOCK_DETECTION:0}",
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
#endif
            "expect_connecting_localities = "
            "${PIKA_EXPECT_CONNECTING_LOCALITIES:0}",

            // add placeholders for keys to be added by command line handling
            "os_threads = cores",
            "cores = all",
            "localities = 1",
            "first_pu = 0",
            "runtime_mode = console",
            "scheduler = local-priority-fifo",
            "affinity = core",
            "pu_step = 1",
            "pu_offset = 0",
            "numa_sensitive = 0",
            "max_background_threads = "
            "${PIKA_MAX_BACKGROUND_THREADS:$[pika.os_threads]}",
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

        /// If PIKA_HAVE_ATTACH_DEBUGGER_ON_TEST_FAILURE is set,
        /// then apply the test-failure value as default.
#if defined(PIKA_HAVE_ATTACH_DEBUGGER_ON_TEST_FAILURE)
            "attach_debugger = ${PIKA_ATTACH_DEBUGGER:test-failure}",
#else
            "attach_debugger = ${PIKA_ATTACH_DEBUGGER}",
#endif
            "exception_verbosity = ${PIKA_EXCEPTION_VERBOSITY:2}",
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
            "use_guard_pages = ${PIKA_USE_GUARD_PAGES:1}",
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

            "[pika.commandline]",
            // enable aliasing
            "aliasing = ${PIKA_COMMANDLINE_ALIASING:1}",

            // allow for unknown options to be passed through
            "allow_unknown = ${PIKA_COMMANDLINE_ALLOW_UNKNOWN:0}",

            // allow for command line options to to be passed through the
            // environment
            "prepend_options = ${PIKA_COMMANDLINE_OPTIONS}",

            // predefine command line aliases
            "[pika.commandline.aliases]",
            "-h = --pika:help",
            "-I = --pika:ini",
            "-p = --pika:app-config",
            "-q = --pika:queuing",
            "-t = --pika:threads",
            "-v = --pika:version",
            "-x = --pika:pika",
            "-0 = --pika:node=0",
            "-1 = --pika:node=1",
            "-2 = --pika:node=2",
            "-3 = --pika:node=3",
            "-4 = --pika:node=4",
            "-5 = --pika:node=5",
            "-6 = --pika:node=6",
            "-7 = --pika:node=7",
            "-8 = --pika:node=8",
            "-9 = --pika:node=9",
            // clang-format on
        };

        lines.insert(lines.end(), extra_static_ini_defs.begin(),
            extra_static_ini_defs.end());

        // don't overload user overrides
        this->parse("<static defaults>", lines, false, false, false);

        need_to_call_pre_initialize = false;
    }

    void runtime_configuration::post_initialize_ini(std::string& pika_ini_file_,
        std::vector<std::string> const& cmdline_ini_defs_)
    {
        util::init_ini_data_base(*this, pika_ini_file_);
        need_to_call_pre_initialize = true;

        // let the command line override the config file.
        if (!cmdline_ini_defs_.empty())
        {
            // do not weed out comments
            this->parse(
                "<command line definitions>", cmdline_ini_defs_, true, false);
            need_to_call_pre_initialize = true;
        }
    }

    void runtime_configuration::pre_initialize_logging_ini()
    {
#if defined(PIKA_HAVE_LOGGING)
        std::vector<std::string> lines = {
        // clang-format off
#define PIKA_TIMEFORMAT "$hh:$mm.$ss.$mili"
#define PIKA_LOGFORMAT "(T%locality%/%pikathread%.%pikaphase%) "

            // general logging
            "[pika.logging]",
            "level = ${PIKA_LOGLEVEL:0}",
            "destination = ${PIKA_LOGDESTINATION:console}",
            "format = ${PIKA_LOGFORMAT:" PIKA_LOGFORMAT
                "P%parentloc%/%pikaparent%.%pikaparentphase% %time%("
                PIKA_TIMEFORMAT ") [%idx%]|\\n}",

            // general console logging
            "[pika.logging.console]",
            "level = ${PIKA_LOGLEVEL:$[pika.logging.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
            "destination = ${PIKA_CONSOLE_LOGDESTINATION:android_log}",
#else
            "destination = ${PIKA_CONSOLE_LOGDESTINATION:"
                "file(pika.$[system.pid].log)}",
#endif
            "format = ${PIKA_CONSOLE_LOGFORMAT:|}",

            // logging related to timing
            "[pika.logging.timing]",
            "level = ${PIKA_TIMING_LOGLEVEL:-1}",
            "destination = ${PIKA_TIMING_LOGDESTINATION:console}",
            "format = ${PIKA_TIMING_LOGFORMAT:" PIKA_LOGFORMAT
                "P%parentloc%/%pikaparent%.%pikaparentphase% %time%("
                PIKA_TIMEFORMAT ") [%idx%] [TIM] |\\n}",

            // console logging related to timing
            "[pika.logging.console.timing]",
            "level = ${PIKA_TIMING_LOGLEVEL:$[pika.logging.timing.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
            "destination = ${PIKA_CONSOLE_TIMING_LOGDESTINATION:android_log}",
#else
            "destination = ${PIKA_CONSOLE_TIMING_LOGDESTINATION:"
                "file(pika.timing.$[system.pid].log)}",
#endif
            "format = ${PIKA_CONSOLE_TIMING_LOGFORMAT:|}",

            // logging related to applications
            "[pika.logging.application]",
            "level = ${PIKA_APP_LOGLEVEL:-1}",
            "destination = ${PIKA_APP_LOGDESTINATION:console}",
            "format = ${PIKA_APP_LOGFORMAT:" PIKA_LOGFORMAT
                "P%parentloc%/%pikaparent%.%pikaparentphase% %time%("
                PIKA_TIMEFORMAT ") [%idx%] [APP] |\\n}",

            // console logging related to applications
            "[pika.logging.console.application]",
            "level = ${PIKA_APP_LOGLEVEL:$[pika.logging.application.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
            "destination = ${PIKA_CONSOLE_APP_LOGDESTINATION:android_log}",
#else
            "destination = ${PIKA_CONSOLE_APP_LOGDESTINATION:"
                "file(pika.application.$[system.pid].log)}",
#endif
            "format = ${PIKA_CONSOLE_APP_LOGFORMAT:|}",

            // logging of debug channel
            "[pika.logging.debuglog]",
            "level = ${PIKA_DEB_LOGLEVEL:-1}",
            "destination = ${PIKA_DEB_LOGDESTINATION:console}",
            "format = ${PIKA_DEB_LOGFORMAT:" PIKA_LOGFORMAT
                "P%parentloc%/%pikaparent%.%pikaparentphase% %time%("
                PIKA_TIMEFORMAT ") [%idx%] [DEB] |\\n}",

            "[pika.logging.console.debuglog]",
            "level = ${PIKA_DEB_LOGLEVEL:$[pika.logging.debuglog.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
            "destination = ${PIKA_CONSOLE_DEB_LOGDESTINATION:android_log}",
#else
            "destination = ${PIKA_CONSOLE_DEB_LOGDESTINATION:"
                "file(pika.debuglog.$[system.pid].log)}",
#endif
            "format = ${PIKA_CONSOLE_DEB_LOGFORMAT:|}"

#undef PIKA_TIMEFORMAT
#undef PIKA_LOGFORMAT
            // clang-format on
        };

        // don't overload user overrides
        this->parse("<static logging defaults>", lines, false, false);
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    runtime_configuration::runtime_configuration(char const* argv0_,
        runtime_mode mode,
        std::vector<std::string> const& extra_static_ini_defs_)
      : extra_static_ini_defs(extra_static_ini_defs_)
      , mode_(mode)
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
        pre_initialize_ini();

        // set global config options
#if PIKA_HAVE_ITTNOTIFY != 0
        use_ittnotify_api = get_itt_notify_mode();
#endif
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

    void runtime_configuration::reconfigure(
        std::vector<std::string> const& cmdline_ini_defs_)
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
#if PIKA_HAVE_ITTNOTIFY != 0
        use_ittnotify_api = get_itt_notify_mode();
#endif
        PIKA_ASSERT(init_small_stack_size() >= PIKA_SMALL_STACK_SIZE);

        small_stacksize = init_small_stack_size();
        medium_stacksize = init_medium_stack_size();
        large_stacksize = init_large_stack_size();
        huge_stacksize = init_huge_stack_size();
    }

    bool runtime_configuration::get_itt_notify_mode() const
    {
#if PIKA_HAVE_ITTNOTIFY != 0
        if (util::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::util::get_entry_as<int>(*sec, "use_itt_notify", 0) != 0;
        }
#endif
        return false;
    }

    // Enable lock detection during suspension
    bool runtime_configuration::enable_lock_detection() const
    {
#ifdef PIKA_HAVE_VERIFY_LOCKS
        if (util::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::util::get_entry_as<int>(*sec, "lock_detection", 0) != 0;
        }
#endif
        return false;
    }

    // Enable minimal deadlock detection for pika threads
    bool runtime_configuration::enable_minimal_deadlock_detection() const
    {
#ifdef PIKA_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
        if (util::section const* sec = get_section("pika"); nullptr != sec)
        {
#ifdef PIKA_DEBUG
            return pika::util::get_entry_as<int>(
                       *sec, "minimal_deadlock_detection", 1) != 0;
#else
            return pika::util::get_entry_as<int>(
                       *sec, "minimal_deadlock_detection", 0) != 0;
#endif
        }

#ifdef PIKA_DEBUG
        return true;
#else
        return false;
#endif

#else
        return false;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_configuration::enable_spinlock_deadlock_detection() const
    {
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
        if (util::section const* sec = get_section("pika"); nullptr != sec)
        {
#ifdef PIKA_DEBUG
            return pika::util::get_entry_as<int>(
                       *sec, "spinlock_deadlock_detection", 1) != 0;
#else
            return pika::util::get_entry_as<int>(
                       *sec, "spinlock_deadlock_detection", 0) != 0;
#endif
        }

#ifdef PIKA_DEBUG
        return true;
#else
        return false;
#endif

#else
        return false;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t runtime_configuration::get_spinlock_deadlock_detection_limit()
        const
    {
#ifdef PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION
        if (util::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::util::get_entry_as<std::size_t>(*sec,
                "spinlock_deadlock_detection_limit",
                PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT);
        }
        return PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT;
#else
        return std::size_t(-1);
#endif
    }

    std::size_t runtime_configuration::trace_depth() const
    {
        if (util::section const* sec = get_section("pika"); nullptr != sec)
        {
            return pika::util::get_entry_as<std::size_t>(
                *sec, "trace_depth", PIKA_HAVE_THREAD_BACKTRACE_DEPTH);
        }
        return PIKA_HAVE_THREAD_BACKTRACE_DEPTH;
    }

    std::size_t runtime_configuration::get_os_thread_count() const
    {
        if (num_os_threads == 0)
        {
            num_os_threads = 1;
            if (util::section const* sec = get_section("pika"); nullptr != sec)
            {
                num_os_threads = pika::util::get_entry_as<std::uint32_t>(
                    *sec, "os_threads", 1);
            }
        }
        return static_cast<std::size_t>(num_os_threads);
    }

    std::string runtime_configuration::get_cmd_line() const
    {
        if (util::section const* sec = get_section("pika"); nullptr != sec)
        {
            return sec->get_entry("cmd_line", "");
        }
        return "";
    }

    // Return the configured sizes of any of the know thread pools
    std::size_t runtime_configuration::get_thread_pool_size(
        char const* poolname) const
    {
        if (util::section const* sec = get_section("pika.threadpools");
            nullptr != sec)
        {
            return pika::util::get_entry_as<std::size_t>(
                *sec, std::string(poolname) + "_size", 2);
        }
        return 2;    // the default size for all pools is 2
    }

    // Will return the stack size to use for all pika-threads.
    std::ptrdiff_t runtime_configuration::init_stack_size(char const* entryname,
        char const* defaultvaluestr, std::ptrdiff_t defaultvalue) const
    {
        if (util::section const* sec = get_section("pika.stacks");
            nullptr != sec)
        {
            std::string entry = sec->get_entry(entryname, defaultvaluestr);
            char* endptr = nullptr;
            std::ptrdiff_t val =
                std::strtoll(entry.c_str(), &endptr, /*base:*/ 0);
            return endptr != entry.c_str() ? val : defaultvalue;
        }
        return defaultvalue;
    }

#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__)
    bool runtime_configuration::use_stack_guard_pages() const
    {
        if (util::section const* sec = get_section("pika.stacks");
            nullptr != sec)
        {
            return pika::util::get_entry_as<int>(*sec, "use_guard_pages", 1) !=
                0;
        }
        return true;    // default is true
    }
#endif

    std::ptrdiff_t runtime_configuration::init_small_stack_size() const
    {
        return init_stack_size("small_size",
            PIKA_PP_STRINGIZE(PIKA_SMALL_STACK_SIZE), PIKA_SMALL_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_medium_stack_size() const
    {
        return init_stack_size("medium_size",
            PIKA_PP_STRINGIZE(PIKA_MEDIUM_STACK_SIZE), PIKA_MEDIUM_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_large_stack_size() const
    {
        return init_stack_size("large_size",
            PIKA_PP_STRINGIZE(PIKA_LARGE_STACK_SIZE), PIKA_LARGE_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_huge_stack_size() const
    {
        return init_stack_size("huge_size",
            PIKA_PP_STRINGIZE(PIKA_HUGE_STACK_SIZE), PIKA_HUGE_STACK_SIZE);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_configuration::load_application_configuration(
        char const* filename, error_code& ec)
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
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what(), pika::rethrow);
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::ptrdiff_t runtime_configuration::get_stack_size(
        threads::thread_stacksize stacksize) const
    {
        switch (stacksize)
        {
        case threads::thread_stacksize::medium:
            return medium_stacksize;

        case threads::thread_stacksize::large:
            return large_stacksize;

        case threads::thread_stacksize::huge:
            return huge_stacksize;

        case threads::thread_stacksize::nostack:
            return (std::numeric_limits<std::ptrdiff_t>::max)();

        default:
        case threads::thread_stacksize::small_:
            break;
        }
        return small_stacksize;
    }
}}    // namespace pika::util
