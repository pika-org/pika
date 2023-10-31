//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/chrono.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

std::uint64_t threshold = 2;

PIKA_NOINLINE std::uint64_t fibonacci_serial(std::uint64_t n)
{
    if (n < 2) return n;
    return fibonacci_serial(n - 1) + fibonacci_serial(n - 2);
}

std::uint64_t add(std::uint64_t n1, std::uint64_t n2) { return n1 + n2; }

///////////////////////////////////////////////////////////////////////////////
ex::unique_any_sender<std::uint64_t> fibonacci_sender_one(std::uint64_t n)
{
    // if we know the answer, we return a sender encapsulating the final value
    if (n < 2) return ex::just(n);
    if (n < threshold) return ex::just(fibonacci_serial(n));

    // asynchronously launch the calculation of one of the sub-terms
    // attach a continuation to this sender which is called asynchronously on
    // its completion and which calculates the other sub-term
    return ex::transfer_just(ex::thread_pool_scheduler{}, n - 1) |
        ex::let_value([n](std::uint64_t res) {
            return ex::when_all(fibonacci_sender_one(n - 2), ex::just(res)) |
                ex::transfer(ex::thread_pool_scheduler{}) | ex::then(add);
        });
}

///////////////////////////////////////////////////////////////////////////////
std::uint64_t fibonacci(std::uint64_t n)
{
    // if we know the answer, we return the final value
    if (n < 2) return n;
    if (n < threshold) return fibonacci_serial(n);

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    auto s = ex::transfer_just(ex::thread_pool_scheduler{}, n - 1) | ex::then(fibonacci);
    std::uint64_t r = fibonacci(n - 2);

    return tt::sync_wait(std::move(s)) + r;
}

///////////////////////////////////////////////////////////////////////////////
ex::unique_any_sender<std::uint64_t> fibonacci_sender(std::uint64_t n)
{
    // if we know the answer, we return a sender encapsulating the final value
    if (n < 2) return ex::just(n);
    if (n < threshold) return ex::just(fibonacci_serial(n));

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    auto s =
        ex::transfer_just(ex::thread_pool_scheduler{}, n - 1) | ex::let_value(fibonacci_sender);
    auto r = fibonacci_sender(n - 2);

    return ex::when_all(std::move(s), std::move(r)) | ex::transfer(ex::thread_pool_scheduler{}) |
        ex::then(add);
}

/////////////////////////////////////////////////////////////////////////////
ex::unique_any_sender<std::uint64_t> fibonacci_sender_all(std::uint64_t n)
{
    // if we know the answer, we return a sender encapsulating the final value
    if (n < 2) return ex::just(n);
    if (n < threshold) return ex::just(fibonacci_serial(n));

    // asynchronously launch the calculation of both of the sub-terms
    auto s =
        ex::transfer_just(ex::thread_pool_scheduler{}, n - 1) | ex::let_value(fibonacci_sender_all);
    auto r =
        ex::transfer_just(ex::thread_pool_scheduler{}, n - 2) | ex::let_value(fibonacci_sender_all);

    // create a sender representing the successful calculation of both sub-terms
    return ex::when_all(std::move(s), std::move(r)) | ex::transfer(ex::thread_pool_scheduler{}) |
        ex::then(add);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    pika::scoped_finalize f;

    // extract command line argument, i.e. fib(N)
    std::uint64_t n = vm["n-value"].as<std::uint64_t>();
    std::string test = vm["test"].as<std::string>();
    std::uint64_t max_runs = vm["n-runs"].as<std::uint64_t>();

    if (max_runs == 0)
    {
        std::cerr << "fibonacci_senders: wrong command line argument value for option 'n-runs', "
                     "should not be zero"
                  << std::endl;
        return -1;
    }

    threshold = vm["threshold"].as<unsigned int>();
    if (threshold < 2 || threshold > n)
    {
        std::cerr << "fibonacci_senders: wrong command line argument value for option 'threshold', "
                     "should be in between 2 and n-value, value specified: "
                  << threshold << std::endl;
        return -1;
    }

    bool executed_one = false;
    std::uint64_t r = 0;

    using namespace std::chrono;
    if (test == "all" || test == "0")
    {
        // Keep track of the time required to execute.
        auto start = high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a sender for the whole calculation, execute it locally,
            // and wait for it.
            r = fibonacci_serial(n);
        }

        std::uint64_t d = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
        constexpr char const* fmt = "fibonacci_serial({}) == {},elapsed time:,{},[ns]\n";
        fmt::print(std::cout, fmt, n, r, d / max_runs);

        executed_one = true;
    }

    if (test == "all" || test == "1")
    {
        // Keep track of the time required to execute.
        auto start = high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a sender for the whole calculation, execute it locally,
            // and wait for it.
            r = tt::sync_wait(fibonacci_sender_one(n));
        }

        std::uint64_t d = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
        constexpr char const* fmt = "fibonacci_sender_one({}) == {},elapsed time:,{},[ns]\n";
        fmt::print(std::cout, fmt, n, r, d / max_runs);

        executed_one = true;
    }

    if (test == "all" || test == "2")
    {
        // Keep track of the time required to execute.
        auto start = high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a sender for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci(n);
        }

        std::uint64_t d = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
        constexpr char const* fmt = "fibonacci({}) == {},elapsed time:,{},[ns]\n";
        fmt::print(std::cout, fmt, n, r, d / max_runs);

        executed_one = true;
    }

    if (test == "all" || test == "3")
    {
        // Keep track of the time required to execute.
        auto start = high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a sender for the whole calculation, execute it locally, and
            // wait for it.
            r = tt::sync_wait(fibonacci_sender(n));
        }

        std::uint64_t d = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
        constexpr char const* fmt = "fibonacci_sender({}) == {},elapsed time:,{},[ns]\n";
        fmt::print(std::cout, fmt, n, r, d / max_runs);

        executed_one = true;
    }

    if (test == "all" || test == "4")
    {
        // Keep track of the time required to execute.
        auto start = high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a sender for the whole calculation, execute it locally, and
            // wait for it.
            r = tt::sync_wait(fibonacci_sender_all(n));
        }

        std::uint64_t d = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
        constexpr char const* fmt = "fibonacci_sender_all({}) == {},elapsed time:,{},[ns]\n";
        fmt::print(std::cout, fmt, n, r, d / max_runs);

        executed_one = true;
    }

    if (!executed_one)
    {
        std::cerr << "fibonacci_senders: wrong command line argument value for option 'tests', "
                     "should be either 'all' or a number between 0 and 4, value specified: "
                  << test << std::endl;
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    pika::program_options::options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    using pika::program_options::value;
    // clang-format off
    desc_commandline.add_options()
        ("n-value", value<std::uint64_t>()->default_value(10),
         "n value for the Fibonacci function")
        ("n-runs", value<std::uint64_t>()->default_value(1),
         "number of runs to perform")
        ("threshold", value<unsigned int>()->default_value(2),
         "threshold for switching to serial code")
        ("test", value<std::string>()->default_value("all"),
        "select tests to execute (0-4, default: all)");
    // clang-format on

    // Initialize and run pika
    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return pika::init(pika_main, argc, argv, init_args);
}
