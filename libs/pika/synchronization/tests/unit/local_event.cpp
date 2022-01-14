//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/runtime_local.hpp>
#include <pika/modules/synchronization.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::threads::get_thread_count;

using pika::this_thread::suspend;

using pika::lcos::local::event;

using pika::local::finalize;
using pika::local::init;

using pika::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
void local_event_test(event& b, std::atomic<std::size_t>& c)
{
    ++c;
    // Wait for the event to occur.
    b.wait();
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    std::size_t pxthreads = 0;

    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();

    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        event e;

        std::atomic<std::size_t> c(0);

        std::vector<pika::future<void>> futs;
        futs.reserve(pxthreads);
        // Create the threads which will wait on the event
        for (std::size_t i = 0; i < pxthreads; ++i)
        {
            futs.push_back(
                pika::async(&local_event_test, std::ref(e), std::ref(c)));
        }

        // Release all the threads.
        e.set();

        // Wait for all the our threads to finish executing.
        pika::wait_all(futs);

        PIKA_TEST_EQ(pxthreads, c);

        // Make sure that waiting on a set event works.
        e.wait();
    }

    // Initiate shutdown of the runtime system.
    return finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("pxthreads,T",
        value<std::size_t>()->default_value(64),
        "the number of PX threads to invoke")("iterations",
        value<std::size_t>()->default_value(64),
        "the number of times to repeat the test");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");
    return report_errors();
}
