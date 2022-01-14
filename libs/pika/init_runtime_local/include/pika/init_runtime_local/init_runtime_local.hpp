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

#include <pika/local/config.hpp>
#include <pika/functional/bind.hpp>
#include <pika/functional/bind_back.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/resource_partitioner/partitioner.hpp>
#include <pika/runtime_local/runtime_local.hpp>
#include <pika/runtime_local/shutdown_function.hpp>
#include <pika/runtime_local/startup_function.hpp>

#include <csignal>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(PIKA_APPLICATION_NAME_DEFAULT) && !defined(PIKA_APPLICATION_NAME)
#define PIKA_APPLICATION_NAME PIKA_APPLICATION_NAME_DEFAULT
#endif

#if !defined(PIKA_APPLICATION_STRING)
#if defined(PIKA_APPLICATION_NAME)
#define PIKA_APPLICATION_STRING PIKA_PP_STRINGIZE(PIKA_APPLICATION_NAME)
#else
#define PIKA_APPLICATION_STRING "unknown pika application"
#endif
#endif

#if defined(__FreeBSD__)
extern PIKA_EXPORT char** freebsd_environ;
extern char** environ;
#endif

namespace pika {
    namespace detail {
        PIKA_EXPORT int init_helper(pika::program_options::variables_map&,
            util::function_nonser<int(int, char**)> const&);
    }

    namespace local {
        namespace detail {
            struct dump_config
            {
                dump_config(pika::runtime const& rt)
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

                std::reference_wrapper<pika::runtime const> rt_;
            };

            // Default params to initialize the init_params struct
            PIKA_MAYBE_UNUSED static int dummy_argc = 1;
            PIKA_MAYBE_UNUSED static char app_name[] = PIKA_APPLICATION_STRING;
            static char* default_argv[2] = {app_name, nullptr};
            PIKA_MAYBE_UNUSED static char** dummy_argv = default_argv;
            // PIKA_APPLICATION_STRING is specific to an application and therefore
            // cannot be in the source file
            PIKA_MAYBE_UNUSED static const pika::program_options::
                options_description default_desc =
                    pika::program_options::options_description(
                        "Usage: " PIKA_APPLICATION_STRING " [options]");

            // Utilities to init the thread_pools of the resource partitioner
            using rp_callback_type =
                pika::util::function_nonser<void(pika::resource::partitioner&,
                    pika::program_options::variables_map const&)>;
        }    // namespace detail

        struct init_params
        {
            std::reference_wrapper<
                pika::program_options::options_description const>
                desc_cmdline = detail::default_desc;
            std::vector<std::string> cfg;
            mutable startup_function_type startup;
            mutable shutdown_function_type shutdown;
            pika::resource::partitioner_mode rp_mode =
                ::pika::resource::mode_default;
            pika::local::detail::rp_callback_type rp_callback;
        };

        namespace detail {
            PIKA_EXPORT int run_or_start(
                util::function_nonser<int(
                    pika::program_options::variables_map& vm)> const& f,
                int argc, char** argv, init_params const& params,
                bool blocking);

            inline int init_start_impl(
                util::function_nonser<int(pika::program_options::variables_map&)>
                    f,
                int argc, char** argv, init_params const& params, bool blocking)
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
                std::signal(SIGABRT, pika::detail::on_abort);
                std::atexit(pika::detail::on_exit);
#if defined(PIKA_HAVE_CXX11_STD_QUICK_EXIT)
                std::at_quick_exit(pika::detail::on_exit);
#endif
                return run_or_start(f, argc, argv, params, blocking);
            }
        }    // namespace detail

        inline int init(
            std::function<int(pika::program_options::variables_map&)> f,
            int argc, char** argv, init_params const& params = init_params())
        {
            return detail::init_start_impl(
                PIKA_MOVE(f), argc, argv, params, true);
        }

        inline int init(std::function<int(int, char**)> f, int argc,
            char** argv, init_params const& params = init_params())
        {
            util::function_nonser<int(pika::program_options::variables_map&)>
                main_f = pika::util::bind_back(pika::detail::init_helper, f);
            return detail::init_start_impl(
                PIKA_MOVE(main_f), argc, argv, params, true);
        }

        inline int init(std::function<int()> f, int argc, char** argv,
            init_params const& params = init_params())
        {
            util::function_nonser<int(pika::program_options::variables_map&)>
                main_f = pika::util::bind(f);
            return detail::init_start_impl(
                PIKA_MOVE(main_f), argc, argv, params, true);
        }

        inline int init(std::nullptr_t, int argc, char** argv,
            init_params const& params = init_params())
        {
            util::function_nonser<int(pika::program_options::variables_map&)>
                main_f;
            return detail::init_start_impl(
                PIKA_MOVE(main_f), argc, argv, params, true);
        }

        inline bool start(
            std::function<int(pika::program_options::variables_map&)> f,
            int argc, char** argv, init_params const& params = init_params())
        {
            return 0 ==
                detail::init_start_impl(PIKA_MOVE(f), argc, argv, params, false);
        }

        inline bool start(std::function<int(int, char**)> f, int argc,
            char** argv, init_params const& params = init_params())
        {
            util::function_nonser<int(pika::program_options::variables_map&)>
                main_f = pika::util::bind_back(pika::detail::init_helper, f);
            return 0 ==
                detail::init_start_impl(
                    PIKA_MOVE(main_f), argc, argv, params, false);
        }

        inline bool start(std::function<int()> f, int argc, char** argv,
            init_params const& params = init_params())
        {
            util::function_nonser<int(pika::program_options::variables_map&)>
                main_f = pika::util::bind(f);
            return 0 ==
                detail::init_start_impl(
                    PIKA_MOVE(main_f), argc, argv, params, false);
        }

        inline bool start(std::nullptr_t, int argc, char** argv,
            init_params const& params = init_params())
        {
            util::function_nonser<int(pika::program_options::variables_map&)>
                main_f;
            return 0 ==
                detail::init_start_impl(
                    PIKA_MOVE(main_f), argc, argv, params, false);
        }

        PIKA_EXPORT int finalize(error_code& ec = throws);
        PIKA_EXPORT int stop(error_code& ec = throws);
        PIKA_EXPORT int suspend(error_code& ec = throws);
        PIKA_EXPORT int resume(error_code& ec = throws);
    }    // namespace local
}    // namespace pika
