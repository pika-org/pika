//  Copyright (c) 2014 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/chrono.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/runtime.hpp>
//
#include <iostream>
#include <random>
#include <utility>
#include <vector>

//
// This is a simple example which generates random numbers and returns
// pass or fail from a routine.
// When called by many threads returning a vector of senders - if the user wants to
// reduce the vector of pass/fails into a single pass fail based on a simple
// any fail = !pass rule, then this example shows how to do it.
// The user can experiment with the failure rate to see if the statistics match
// their expectations.
// Also. Routine can use either a lambda, or a function under control of USE_LAMBDA

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

#define TEST_SUCCESS 1
#define TEST_FAIL 0
//
#define FAILURE_RATE_PERCENT 5
#define SAMPLES_PER_LOOP 10
#define TEST_LOOPS 1000
//
std::random_device rseed;
std::mt19937 gen(rseed());
std::uniform_int_distribution<int> dist(0, 99);    // interval [0,100)

#define USE_LAMBDA

//----------------------------------------------------------------------------
int reduce(std::vector<int>&& vec)
{
    for (int t : vec)
    {
        if (t == TEST_FAIL) return TEST_FAIL;
    }
    return TEST_SUCCESS;
}

//----------------------------------------------------------------------------
int generate_one()
{
    // generate roughly x% fails
    int result = TEST_SUCCESS;
    if (dist(gen) >= (100 - FAILURE_RATE_PERCENT)) { result = TEST_FAIL; }
    return result;
}

//----------------------------------------------------------------------------
auto test_reduce()
{
    std::vector<ex::unique_any_sender<int>> req_senders;
    //
    for (int i = 0; i < SAMPLES_PER_LOOP; i++)
    {
        // generate random sequence of pass/fails using % fail rate per incident
        req_senders.push_back(ex::schedule(ex::thread_pool_scheduler{}) | ex::then(generate_one));
    }

    auto all_ready = ex::when_all_vector(std::move(req_senders));

#ifdef USE_LAMBDA
    auto result = std::move(all_ready) | ex::then([](std::vector<int>&& vec) -> int {
        for (int t : vec)
        {
            if (t == TEST_FAIL) return TEST_FAIL;
        }
        return TEST_SUCCESS;
    });
#else
    auto result = std::move(all_ready) | ex::then(reduce);
#endif
    //
    return result;
}

//----------------------------------------------------------------------------
int pika_main()
{
    pika::chrono::detail::high_resolution_timer htimer;
    // run N times and see if we get approximately the right amount of fails
    int count = 0;
    for (int i = 0; i < TEST_LOOPS; i++)
    {
        int result = tt::sync_wait(test_reduce());
        count += result;
    }
    double pr_pass = std::pow(1.0 - FAILURE_RATE_PERCENT / 100.0, SAMPLES_PER_LOOP);
    double exp_pass = TEST_LOOPS * pr_pass;
    std::cout << "From " << TEST_LOOPS << " tests, we got "
              << "\n " << count << " passes"
              << "\n " << exp_pass << " expected \n"
              << "\n " << htimer.elapsed<std::chrono::seconds>() << " seconds \n"
              << std::flush;
    // Initiate shutdown of the runtime system.
    return pika::finalize();
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Initialize and run pika.
    return pika::init(pika_main, argc, argv);
}
