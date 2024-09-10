//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is the second in a series of examples demonstrating the development of a fully distributed
// solver for a simple 1D heat distribution problem.
//
// This example shows how the code from example one can be made asynchronous. While this nicely
// parallelizes the code (note: without changing the overall structure of the algorithm), the
// achieved performance is bad (a lot slower than example one). This is caused by the large amount
// of overheads introduced by wrapping each and every grid point into its own sender The amount of
// work performed by each of the created pika threads (one thread for every grid point and time
// step) is too small compared to the imposed overheads.

#include <pika/assert.hpp>
#include <pika/chrono.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "print_time_results.hpp"

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true;    // print csv heading
double k = 0.5;        // heat transfer coefficient
double dt = 1.;        // time step
double dx = 1.;        // grid spacing

inline std::size_t idx(std::size_t i, int dir, std::size_t size)
{
    if (i == 0 && dir == -1) return size - 1;
    if (i == size - 1 && dir == +1) return 0;

    PIKA_ASSERT((i + dir) < size);

    return i + dir;
}

///////////////////////////////////////////////////////////////////////////////
//[stepper_2
struct stepper
{
    // Our partition type
    using partition = ex::any_sender<double>;

    // Our data for one time step
    using space = std::vector<partition>;

    // Our operator
    static double heat(double left, double middle, double right)
    {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // do all the work on 'nx' data points for 'nt' time steps
    space do_work(std::size_t nx, std::size_t nt)
    {
        auto sched = ex::thread_pool_scheduler{};

        // U[t][i] is the state of position i at time t.
        std::array<space, 2> U{};
        for (space& s : U) s.resize(nx);

        // Initial conditions: f(0, i) = i
        for (std::size_t i = 0; i != nx; ++i) U[0][i] = ex::just(double(i));

        auto Op = stepper::heat;

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t)
        {
            space const& current = U[t % 2];
            space& next = U[(t + 1) % 2];

            // WHEN U[t][i-1], U[t][i], and U[t][i+1] have been computed, THEN we
            // can compute U[t+1][i]
            for (std::size_t i = 0; i != nx; ++i)
            {
                next[i] =
                    ex::when_all(current[idx(i, -1, nx)], current[i], current[idx(i, +1, nx)]) |
                    ex::continues_on(sched) | ex::then(Op) | ex::split();
            }
        }

        // Now the asynchronous computation is running; the above for-loop does not
        // wait on anything. There is no implicit waiting at the end of each timestep;
        // the computation of each U[t][i] will begin as soon as its dependencies
        // are ready and hardware is available.

        // Return the solution at time-step 'nt'.
        return U[nt % 2];
    }
};
//]
///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    std::uint64_t nx = vm["nx"].as<std::uint64_t>();    // Number of grid points.
    std::uint64_t nt = vm["nt"].as<std::uint64_t>();    // Number of steps.

    if (vm.count("no-header")) header = false;

    // Create the stepper object
    stepper step;

    using namespace std::chrono;
    // Measure execution time.
    auto t = high_resolution_clock::now();

    // Execute nt time steps on nx grid points.
    auto solution = tt::sync_wait(ex::when_all_vector(step.do_work(nx, nt)));

    double elapsed = duration<double>(high_resolution_clock::now() - t).count();

    // Print the final solution
    if (vm.count("results"))
    {
        for (std::size_t i = 0; i != nx; ++i)
            std::cout << "U[" << i << "] = " << solution[i] << std::endl;
    }

    std::uint64_t const os_thread_count = pika::get_os_thread_count();
    print_time_results(os_thread_count, elapsed, nx, nt, header);

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    using namespace pika::program_options;

    options_description desc_commandline;
    // clang-format off
    desc_commandline.add_options()
        ("results", "print generated results (default: false)")
        ("nx", value<std::uint64_t>()->default_value(100),
         "Local x dimension")
        ("nt", value<std::uint64_t>()->default_value(45),
         "Number of time steps")
        ("k", value<double>(&k)->default_value(0.5),
         "Heat transfer coefficient (default: 0.5)")
        ("dt", value<double>(&dt)->default_value(1.0),
         "Timestep unit (default: 1.0[s])")
        ("dx", value<double>(&dx)->default_value(1.0),
         "Local x dimension")
        ( "no-header", "do not print out the csv header row")
    ;
    // clang-format on

    // Initialize and run pika
    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return pika::init(pika_main, argc, argv, init_args);
}
