//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/local/functional.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/threadmanager.hpp>
#include <pika/modules/timing.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "worker_timed.hpp"

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::get_os_thread_count;

using pika::threads::make_thread_function_nullary;
using pika::threads::register_work;
using pika::threads::thread_init_data;

using pika::this_thread::suspend;
using pika::threads::get_thread_count;

using pika::chrono::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables.
std::uint64_t tasks = 500000;
std::uint64_t min_delay = 0;
std::uint64_t max_delay = 0;
std::uint64_t total_delay = 0;
std::uint64_t seed = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(std::uint64_t cores, double walltime)
{
    if (header)
        std::cout << "OS-threads,Seed,Tasks,Minimum Delay (iterations),"
                     "Maximum Delay (iterations),Total Delay (iterations),"
                     "Total Walltime (seconds),Walltime per Task (seconds)\n"
                  << std::flush;

    std::string const cores_str = pika::util::format("{},", cores);
    std::string const seed_str = pika::util::format("{},", seed);
    std::string const tasks_str = pika::util::format("{},", tasks);

    std::string const min_delay_str = pika::util::format("{},", min_delay);
    std::string const max_delay_str = pika::util::format("{},", max_delay);
    std::string const total_delay_str = pika::util::format("{},", total_delay);

    pika::util::format_to(std::cout,
        "{:-21} {:-21} {:-21} {:-21} {:-21} {:-21} {:10.12}, {:10.12}\n",
        cores_str, seed_str, tasks_str, min_delay_str, max_delay_str,
        total_delay_str, walltime, walltime / tasks)
        << std::flush;
}

///////////////////////////////////////////////////////////////////////////////
std::uint64_t shuffler(std::mt19937_64& prng, std::uint64_t high)
{
    if (high == 0)
        throw std::logic_error("high value was 0");

    // Our range is [0, x).
    std::uniform_int_distribution<std::uint64_t> dist(0, high - 1);

    return dist(prng);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    if (vm.count("no-header"))
        header = false;

    ///////////////////////////////////////////////////////////////////////////
    // Initialize the PRNG seed.
    if (!seed)
        seed = std::uint64_t(std::time(nullptr));

    {
        ///////////////////////////////////////////////////////////////////////
        // Validate command-line arguments.
        if (0 == tasks)
            throw std::invalid_argument("count of 0 tasks specified\n");

        if (min_delay > max_delay)
            throw std::invalid_argument("minimum delay cannot be larger than "
                                        "maximum delay\n");

        if (min_delay > total_delay)
            throw std::invalid_argument("minimum delay cannot be larger than"
                                        "total delay\n");

        if (max_delay > total_delay)
            throw std::invalid_argument("maximum delay cannot be larger than "
                                        "total delay\n");

        if ((min_delay * tasks) > total_delay)
            throw std::invalid_argument("minimum delay is too small for the "
                                        "specified total delay and number of "
                                        "tasks\n");

        if ((max_delay * tasks) < total_delay)
            throw std::invalid_argument("maximum delay is too small for the "
                                        "specified total delay and number of "
                                        "tasks\n");

        ///////////////////////////////////////////////////////////////////////
        // Randomly generate a description of the heterogeneous workload.
        std::vector<std::uint64_t> payloads;
        payloads.reserve(tasks);

        // For random numbers, we use a 64-bit specialization of stdlib's
        // mersenne twister engine (good uniform distribution up to 311
        // dimensions, cycle length 2 ^ 19937 - 1)
        std::mt19937_64 prng(seed);

        std::uint64_t current_sum = 0;

        for (std::uint64_t i = 0; i < tasks; ++i)
        {
            // Credit to Spencer Ruport for putting this algorithm on
            // stackoverflow.
            std::uint64_t const low_calc =
                (total_delay - current_sum) - (max_delay * (tasks - 1 - i));

            bool const negative =
                (total_delay - current_sum) < (max_delay * (tasks - 1 - i));

            std::uint64_t const low =
                (negative || (low_calc < min_delay)) ? min_delay : low_calc;

            std::uint64_t const high_calc =
                (total_delay - current_sum) - (min_delay * (tasks - 1 - i));

            std::uint64_t const high =
                (high_calc > max_delay) ? max_delay : high_calc;

            // Our range is [low, high].
            std::uniform_int_distribution<std::uint64_t> dist(low, high);

            std::uint64_t const payload = dist(prng);

            if (payload < min_delay)
                throw std::logic_error("task delay is below minimum");

            if (payload > max_delay)
                throw std::logic_error("task delay is above maximum");

            current_sum += payload;
            payloads.push_back(payload);
        }

        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::shuffle(payloads.begin(), payloads.end(), std::move(generator));

        ///////////////////////////////////////////////////////////////////////
        // Validate the payloads.
        if (payloads.size() != tasks)
            throw std::logic_error("incorrect number of tasks generated");

        std::uint64_t const payloads_sum =
            std::accumulate(payloads.begin(), payloads.end(), 0ULL);
        if (payloads_sum != total_delay)
            throw std::logic_error("incorrect total delay generated");

        ///////////////////////////////////////////////////////////////////////
        // Start the clock.
        high_resolution_timer t;

        ///////////////////////////////////////////////////////////////////////
        // Queue the tasks in a serial loop.
        for (std::uint64_t i = 0; i < tasks; ++i)
        {
            thread_init_data data(make_thread_function_nullary(pika::util::bind(
                                      &worker_timed, payloads[i] * 1000)),
                "worker_timed");
            register_work(data);
        }

        ///////////////////////////////////////////////////////////////////////
        // Wait for the work to finish.
        do
        {
            // Reschedule pika_main until all other pika-threads have finished. We
            // should be resumed after most of the null pika-threads have been
            // executed. If we haven't, we just reschedule ourselves again.
            suspend();
        } while (get_thread_count(pika::threads::thread_priority::normal) > 1);

        ///////////////////////////////////////////////////////////////////////
        // Print the results.
        print_results(get_os_thread_count(), t.elapsed());
    }

    pika::local::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ( "tasks"
        , value<std::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks to invoke")

        ( "min-delay"
        , value<std::uint64_t>(&min_delay)->default_value(0)
        , "minimum number of iterations in the delay loop")

        ( "max-delay"
        , value<std::uint64_t>(&max_delay)->default_value(0)
        , "maximum number of iterations in the delay loop")

        ( "total-delay"
        , value<std::uint64_t>(&total_delay)->default_value(0)
        , "total number of delay iterations to be executed")

        ( "seed"
        , value<std::uint64_t>(&seed)->default_value(0)
        , "seed for the pseudo random number generator (if 0, a seed is "
          "chosen based on the current system time)")

        ( "no-header"
        , "do not print out the csv header row")
        ;
    // clang-format on

    // Initialize and run pika.
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::local::init(pika_main, argc, argv, init_args);
}
#endif
