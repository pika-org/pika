//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: pikainspect:nodeprecatedinclude pikainspect:nodeprecatedname

#include <pika/functional/function.hpp>
#include <pika/modules/timing.hpp>
#include <pika/pika.hpp>

#include <pika/modules/program_options.hpp>
#include <boost/function.hpp>

#include <cstdint>
#include <functional>
#include <iostream>

#include "worker_timed.hpp"

using pika::program_options::command_line_parser;
using pika::program_options::notify;
using pika::program_options::options_description;
using pika::program_options::store;
using pika::program_options::value;
using pika::program_options::variables_map;

std::uint64_t iterations = 500000;
std::uint64_t delay = 5;

struct foo
{
    void operator()() const
    {
        worker_timed(delay * 1000);
    }
};

template <typename F>
void run(F const& f, std::uint64_t local_iterations)
{
    std::uint64_t i = 0;
    pika::chrono::detail::high_resolution_timer t;

    for (; i < local_iterations; ++i)
        f();

    double elapsed = t.elapsed();
    std::cout << " walltime/iteration: " << ((elapsed / i) * 1e9) << " ns\n";
}

int app_main(variables_map& vm)
{
    {
        foo f;
        std::cout << "baseline";
        run(f, iterations);
    }
    {
        pika::util::detail::function<void(), false> f = foo();
        std::cout << "pika::util::detail::function (non-serializable)";
        run(f, iterations);
    }
    {
        pika::util::detail::function<void()> f = foo();
        std::cout << "pika::util::detail::function (serializable)";
        run(f, iterations);
    }
    {
        boost::function<void()> f = foo();
        std::cout << "boost::function";
        run(f, iterations);
    }
    {
        std::function<void()> f = foo();
        std::cout << "std::function";
        run(f, iterations);
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    // Parse command line.
    variables_map vm;

    options_description cmdline("Usage: " PIKA_APPLICATION_STRING " [options]");

    cmdline.add_options()("help,h", "print out program usage (this message)")

        ("iterations", value<std::uint64_t>(&iterations)->default_value(500000),
            "number of iterations to invoke for each test")

            ("delay", value<std::uint64_t>(&delay)->default_value(5),
                "duration of delay in microseconds");

    store(command_line_parser(argc, argv).options(cmdline).run(), vm);

    notify(vm);

    // Print help screen.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        return 0;
    }

    return app_main(vm);
}
