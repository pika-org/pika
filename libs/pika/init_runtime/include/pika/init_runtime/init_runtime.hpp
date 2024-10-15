//  Copyright (c)      2021 ETH Zurich
//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/bind.hpp>
#include <pika/functional/bind_back.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/resource_partitioner/partitioner.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/shutdown_function.hpp>
#include <pika/runtime/startup_function.hpp>

#include <csignal>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(PIKA_APPLICATION_NAME_DEFAULT) && !defined(PIKA_APPLICATION_NAME)
# define PIKA_APPLICATION_NAME PIKA_APPLICATION_NAME_DEFAULT
#endif

#if !defined(PIKA_APPLICATION_STRING)
# if defined(PIKA_APPLICATION_NAME)
#  define PIKA_APPLICATION_STRING PIKA_PP_STRINGIZE(PIKA_APPLICATION_NAME)
# else
#  define PIKA_APPLICATION_STRING "unknown pika application"
# endif
#endif

#if defined(__FreeBSD__)
extern PIKA_EXPORT char** freebsd_environ;
extern char** environ;
#endif

namespace pika {
    namespace detail {
        // Default params to initialize the init_params struct
        [[maybe_unused]] static int dummy_argc = 1;
        [[maybe_unused]] static char app_name[] = PIKA_APPLICATION_STRING;
        static char* default_argv[2] = {app_name, nullptr};
        [[maybe_unused]] static char** dummy_argv = default_argv;
        // PIKA_APPLICATION_STRING is specific to an application and therefore
        // cannot be in the source file
        [[maybe_unused]] static const pika::program_options::options_description default_desc =
            pika::program_options::options_description(
                "Usage: " PIKA_APPLICATION_STRING " [options]");

        // Utilities to init the thread_pools of the resource partitioner
        using rp_callback_type = pika::util::detail::function<void(
            pika::resource::partitioner&, pika::program_options::variables_map const&)>;
    }    // namespace detail

    struct init_params
    {
        std::reference_wrapper<pika::program_options::options_description const> desc_cmdline =
            detail::default_desc;
        std::vector<std::string> cfg;
        mutable startup_function_type startup;
        mutable shutdown_function_type shutdown;
        pika::resource::partitioner_mode rp_mode = pika::resource::mode_default;
        detail::rp_callback_type rp_callback;
    };

    PIKA_EXPORT int init(std::function<int(pika::program_options::variables_map&)> f, int argc,
        const char* const* argv, init_params const& params = init_params());
    PIKA_EXPORT int init(std::function<int(int, char**)> f, int argc, const char* const* argv,
        init_params const& params = init_params());
    PIKA_EXPORT int init(std::function<int()> f, int argc, const char* const* argv,
        init_params const& params = init_params());
    PIKA_EXPORT int init(std::nullptr_t, int argc, const char* const* argv,
        init_params const& params = init_params());

    /// Start the runtime.
    ///
    /// @param f entry point of the first task on the pika runtime. f will be passed all non-pika
    /// command line arguments.
    /// @param argc number of arguments in argv
    /// @param argv array of arguments. The first element is ignored.
    ///
    /// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
    /// @pre the runtime is stopped
    /// @post the runtime is running
    PIKA_EXPORT void start(std::function<int(pika::program_options::variables_map&)> f, int argc,
        const char* const* argv, init_params const& params = init_params());

    /// Start the runtime.
    ///
    /// @param f entry point of the first task on the pika runtime. f will be passed all non-pika
    /// command line arguments.
    /// @param argc number of arguments in argv
    /// @param argv array of arguments. The first element is ignored.
    ///
    /// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
    /// @pre the runtime is stopped
    /// @post the runtime is running
    PIKA_EXPORT void start(std::function<int(int, char**)> f, int argc, const char* const* argv,
        init_params const& params = init_params());

    /// Start the runtime.
    ///
    /// @param f entry point of the first task on the pika runtime
    /// @param argc number of arguments in argv
    /// @param argv array of arguments. The first element is ignored.
    ///
    /// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
    /// @pre the runtime is not running
    PIKA_EXPORT void start(std::function<int()> f, int argc, const char* const* argv,
        init_params const& params = init_params());

    PIKA_EXPORT void start(std::nullptr_t, int argc, const char* const* argv,
        init_params const& params = init_params());

    /// Start the runtime.
    ///
    /// No task is created on the runtime.
    ///
    /// @param argc number of arguments in argv
    /// @param argv array of arguments. The first element is ignored.
    ///
    /// @pre `(argc == 0 && argv == nullptr) || (argc >= 1 && argv != nullptr)`
    /// @pre the runtime is not initialized
    /// @post the runtime is running
    PIKA_EXPORT void start(
        int argc, const char* const* argv, init_params const& params = init_params());

    /// Stop the runtime.
    ///
    /// Waits until @ref pika::finalize has been called and there is no more activity on the
    /// runtime. See @ref pika::wait. The runtime can be started again after calling @ref
    /// pika::stop. Must be called from outside the runtime.
    ///
    /// @return the return value of the callable passed to @p pika::start, if any. If none was
    /// passed, returns 0.
    ///
    /// @pre the runtime is initialized
    /// @pre the calling thread is not a pika task
    /// @post the runtime is not initialized
    PIKA_EXPORT int stop();

    /// Signal the runtime that it may be stopped.
    ///
    /// Until @ref pika::finalize has been called, @ref pika::stop will not return. This function
    /// exists to distinguish between the runtime being idle but still expecting work to be
    /// scheduled on it and the runtime being idle and ready to be shutdown. Unlike @pika::stop,
    /// @ref pika::finalize can be called from within or outside the runtime.
    ///
    /// @pre the runtime is initialized
    PIKA_EXPORT void finalize();

    /// Wait for the runtime to be idle.
    ///
    /// Waits until the runtime is idle. This includes tasks scheduled on the thread pools as well
    /// as non-tasks such as CUDA kernels submitted through pika facilities. Can be called from
    /// within the runtime, in which case the calling task is ignored when determining idleness.
    ///
    /// @pre the runtime is initialized
    /// @post all work submitted before the call to wait is completed
    PIKA_EXPORT void wait();

    /// Suspend the runtime.
    ///
    /// Waits until the runtime is idle and suspends worker threads on all thread pools. Work can be
    /// scheduled on the runtime even when it is suspended, but no progress will be made.
    ///
    /// @pre the calling thread is not a pika task
    /// @pre runtime is running or suspended
    /// @post runtime is suspended
    PIKA_EXPORT void suspend();

    /// Resume the runtime.
    ///
    /// Resumes the runtime by waking all worker threads on all thread pools.
    ///
    /// @pre the calling thread is not a pika task
    /// @pre runtime is suspended or running
    /// @post runtime is running
    PIKA_EXPORT void resume();
}    // namespace pika
