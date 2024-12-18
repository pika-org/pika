//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/barrier.hpp>
#include <pika/init.hpp>
#include <pika/modules/allocator_support.hpp>
#include <pika/runtime.hpp>
#include <pika/thread.hpp>

#include <boost/lockfree/queue.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>

template <typename T>
using queue = boost::lockfree::queue<T, pika::detail::aligned_allocator<std::size_t>>;

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using barrier = pika::barrier<>;

using pika::threads::detail::register_work;
using pika::threads::detail::thread_init_data;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
std::atomic<double> global_scratch = 0;
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double delay()
{
    double d = 0.;
    for (std::uint64_t i = 0; i < num_iterations; ++i) d += 1 / (2. * static_cast<double>(i) + 1);
    return d;
}

///////////////////////////////////////////////////////////////////////////////
void get_os_thread_num(barrier& barr, queue<std::size_t>& os_threads)
{
    global_scratch.store(delay(), std::memory_order_relaxed);
    os_threads.push(pika::get_worker_thread_num());
    // `arrive_and_drop` is necessary here since the barrier can go out of scope
    // in pika_main.
    barr.arrive_and_drop();
}

///////////////////////////////////////////////////////////////////////////////
using result_map = std::map<std::size_t, std::size_t>;

using sorter = std::multimap<std::size_t, std::size_t, std::greater<std::size_t>>;

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    {
        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        bool const csv = vm.count("csv");

        std::size_t const pxthreads = vm["pxthreads"].as<std::size_t>();

        result_map results;

        {
            // Have the queue preallocate the nodes.
            queue<std::size_t> os_threads(pxthreads);

            barrier barr(pxthreads + 1);

            for (std::size_t j = 0; j < pxthreads; ++j)
            {
                thread_init_data data(
                    pika::threads::detail::make_thread_function_nullary(pika::util::detail::bind(
                        &get_os_thread_num, std::ref(barr), std::ref(os_threads))),
                    "get_os_thread_num", pika::execution::thread_priority::normal,
                    pika::execution::thread_schedule_hint(0));
                register_work(data);
            }

            barr.arrive_and_wait();    // wait for all PX threads to enter the barrier

            std::size_t shepherd = 0;

            while (os_threads.pop(shepherd)) ++results[shepherd];
        }

        sorter sort;

        for (result_map::value_type const& result : results)
        {
            sort.insert(sorter::value_type(result.second, result.first));
        }

        for (sorter::value_type const& result : sort)
        {
            if (csv)
                fmt::print(std::cout, "{},{}\n", result.second, result.first);
            else
                fmt::print(
                    std::cout, "OS-thread {} ran {} PX-threads\n", result.second, result.first);
        }
    }

    // initiate shutdown of the runtime system
    pika::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " PIKA_APPLICATION_STRING " [options]");

    cmdline.add_options()(
        "pxthreads", value<std::size_t>()->default_value(128), "number of PX-threads to invoke")

        ("delay-iterations", value<std::uint64_t>()->default_value(65536),
            "number of iterations in the delay loop")

            ("csv", "output results as csv (format: OS-thread,PX-threads)");

    // Initialize and run pika
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(pika_main, argc, argv, init_args);
}
