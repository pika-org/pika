//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/chrono.hpp>
#include <pika/init.hpp>
#include <pika/thread.hpp>

#include <chrono>
#include <iostream>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using std::chrono::seconds;

using pika::threads::detail::get_self;
using pika::threads::detail::get_self_id;
using pika::threads::detail::set_thread_state;

using pika::chrono::detail::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    {
        std::cout << "waiting for 5 seconds\n";

        high_resolution_timer t;

        // Schedule a wakeup in 5 seconds.
        set_thread_state(
            get_self_id(), seconds(5), pika::threads::detail::thread_schedule_state::pending);

        // Suspend this pika thread.
        pika::this_thread::suspend(pika::threads::detail::thread_schedule_state::suspended);

        std::cout << "woke up after " << t.elapsed<seconds>() << " seconds\n";
    }

    pika::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description desc_commandline("Usage: " PIKA_APPLICATION_STRING " [options]");

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return pika::init(pika_main, argc, argv, init_args);
}
