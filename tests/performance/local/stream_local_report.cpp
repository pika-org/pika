//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// This code is based on the STREAM benchmark:
// https://www.cs.virginia.edu/stream/ref.html
//
// We adopted the code and pikaifyed it.
//

#if defined(PIKA_MSVC_NVCC)
// NVCC causes an ICE in MSVC if this is not defined
#define BOOST_NO_CXX11_ALLOCATOR
#endif

#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/local/version.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/testing.hpp>
#include <pika/type_support/unused.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

bool csv = false;
bool header = false;

///////////////////////////////////////////////////////////////////////////////
std::string get_executor_name(std::size_t executor)
{
    switch (executor)
    {
    case 0:
        return "parallel_executor";
    case 1:
        return "fork_join_executor";
    case 2:
        return "scheduler_executor";
    default:
        return "no-executor";
    }
}

///////////////////////////////////////////////////////////////////////////////
pika::threads::topology& retrieve_topology()
{
    static pika::threads::topology& topo = pika::threads::create_topology();
    return topo;
}

///////////////////////////////////////////////////////////////////////////////
double mysecond()
{
    return pika::chrono::high_resolution_clock::now() * 1e-9;
}

int checktick()
{
    static const std::size_t M = 20;
    double timesfound[M];

    // Collect a sequence of M unique time values from the system.
    for (std::size_t i = 0; i < M; i++)
    {
        double const t1 = mysecond();
        double t2;
        while (((t2 = mysecond()) - t1) < 1.0E-6)
            ;
        timesfound[i] = t2;
    }

    // Determine the minimum difference between these M values.
    // This result will be our estimate (in microseconds) for the
    // clock granularity.
    int minDelta = 1000000;
    for (std::size_t i = 1; i < M; i++)
    {
        int Delta = (int) (1.0E6 * (timesfound[i] - timesfound[i - 1]));
        minDelta = (std::min)(minDelta, (std::max)(Delta, 0));
    }

    return (minDelta);
}

template <typename Vector>
void check_results(std::size_t iterations, Vector const& a_res,
    Vector const& b_res, Vector const& c_res)
{
    std::vector<STREAM_TYPE> a(a_res.size());
    std::vector<STREAM_TYPE> b(b_res.size());
    std::vector<STREAM_TYPE> c(c_res.size());

    pika::copy(pika::execution::par, a_res.begin(), a_res.end(), a.begin());
    pika::copy(pika::execution::par, b_res.begin(), b_res.end(), b.begin());
    pika::copy(pika::execution::par, c_res.begin(), c_res.end(), c.begin());

    STREAM_TYPE aj, bj, cj, scalar;
    STREAM_TYPE aSumErr, bSumErr, cSumErr;
    STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;
    double epsilon;
    int ierr, err;

    /* reproduce initialization */
    aj = 1.0;
    bj = 2.0;
    cj = 0.0;
    /* now execute timing loop */
    scalar = 3.0;
    for (std::size_t k = 0; k < iterations; k++)
    {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    /* accumulate deltas between observed and expected results */
    aSumErr = 0.0;
    bSumErr = 0.0;
    cSumErr = 0.0;
    for (std::size_t j = 0; j < a.size(); j++)
    {
        aSumErr += std::abs(a[j] - aj);
        bSumErr += std::abs(b[j] - bj);
        cSumErr += std::abs(c[j] - cj);
    }
    aAvgErr = aSumErr / (STREAM_TYPE) a.size();
    bAvgErr = bSumErr / (STREAM_TYPE) a.size();
    cAvgErr = cSumErr / (STREAM_TYPE) a.size();

    if (sizeof(STREAM_TYPE) == 4)
    {
        epsilon = 1.e-6;
    }
    else if (sizeof(STREAM_TYPE) == 8)
    {
        epsilon = 1.e-13;
    }
    else
    {
        pika::util::format_to(std::cout, "WEIRD: sizeof(STREAM_TYPE) = {}\n",
            sizeof(STREAM_TYPE));
        epsilon = 1.e-6;
    }

    err = 0;
    if (std::abs(aAvgErr / aj) > epsilon)
    {
        err++;
        pika::util::format_to(std::cout,
            "Failed Validation on array a[], AvgRelAbsErr > epsilon ({})\n",
            epsilon);
        pika::util::format_to(std::cout,
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}\n", aj,
            aAvgErr, std::abs(aAvgErr) / aj);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(a[j] / aj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    pika::util::format_to(std::cout,
                        "         array a: index: {}, expected: {}, "
                        "observed: {}, relative error: {}\n",
                        (unsigned long) j, aj, a[j],
                        (double) std::abs((aj - a[j]) / aAvgErr));
                }
#endif
            }
        }
        pika::util::format_to(
            std::cout, "     For array a[], {} errors were found.\n", ierr);
    }
    if (std::abs(bAvgErr / bj) > epsilon)
    {
        err++;
        pika::util::format_to(std::cout,
            "Failed Validation on array b[], AvgRelAbsErr > epsilon ({})\n",
            epsilon);
        pika::util::format_to(std::cout,
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}\n", bj,
            bAvgErr, std::abs(bAvgErr) / bj);
        pika::util::format_to(
            std::cout, "     AvgRelAbsErr > Epsilon ({})\n", epsilon);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(b[j] / bj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    pika::util::format_to(std::cout,
                        "         array b: index: {}, expected: {}, "
                        "observed: {}, relative error: {}\n",
                        (unsigned long) j, bj, b[j],
                        (double) std::abs((bj - b[j]) / bAvgErr));
                }
#endif
            }
        }
        pika::util::format_to(
            std::cout, "     For array b[], {} errors were found.\n", ierr);
    }
    if (std::abs(cAvgErr / cj) > epsilon)
    {
        err++;
        pika::util::format_to(std::cout,
            "Failed Validation on array c[], AvgRelAbsErr > epsilon ({})\n",
            epsilon);
        pika::util::format_to(std::cout,
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}\n", cj,
            cAvgErr, std::abs(cAvgErr) / cj);
        pika::util::format_to(
            std::cout, "     AvgRelAbsErr > Epsilon ({})\n", epsilon);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(c[j] / cj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    pika::util::format_to(std::cout,
                        "         array c: index: {}, expected: {}, "
                        "observed: {}, relative error: {}\n",
                        (unsigned long) j, cj, c[j],
                        (double) std::abs((cj - c[j]) / cAvgErr));
                }
#endif
            }
        }
        pika::util::format_to(
            std::cout, "     For array c[], {} errors were found.\n", ierr);
    }
    if (err == 0)
    {
        if (!csv)
        {
            pika::util::format_to(std::cout,
                "Solution Validates: avg error less than {} on all three "
                "arrays\n",
                epsilon);
        }
    }
#ifdef VERBOSE
    pika::util::format_to(std::cout, "Results Validation Verbose Results:\n");
    pika::util::format_to(
        std::cout, "    Expected a(1), b(1), c(1): {} {} {}\n", aj, bj, cj);
    pika::util::format_to(std::cout, "    Observed a(1), b(1), c(1): {} {} {}\n",
        a[1], b[1], c[1]);
    pika::util::format_to(std::cout, "    Rel Errors on a, b, c:     {} {} {}\n",
        (double) std::abs(aAvgErr / aj), (double) std::abs(bAvgErr / bj),
        (double) std::abs(cAvgErr / cj));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct multiply_step
{
    explicit multiply_step(T factor)
      : factor_(factor)
    {
    }

    // FIXME : call operator of multiply_step is momentarily defined with
    //         a generic parameter to allow the host_side invoke_result<>
    //         (used in invoke()) to get the return type

    template <typename U>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE T operator()(U val) const
    {
        return val * factor_;
    }

    T factor_;
};

template <typename T>
struct add_step
{
    // FIXME : call operator of add_step is momentarily defined with
    //         generic parameters to allow the host_side invoke_result<>
    //         (used in invoke()) to get the return type

    template <typename U>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE T operator()(U val1, U val2) const
    {
        return val1 + val2;
    }
};

template <typename T>
struct triad_step
{
    explicit triad_step(T factor)
      : factor_(factor)
    {
    }

    // FIXME : call operator of triad_step is momentarily defined with
    //         generic parameters to allow the host_side invoke_result<>
    //         (used in invoke()) to get the return type

    template <typename U>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE T operator()(U val1, U val2) const
    {
        return val1 + val2 * factor_;
    }

    T factor_;
};

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
auto run_benchmark(std::size_t warmup_iterations, std::size_t iterations,
    std::size_t size, Policy&& policy, std::size_t executor)
{
    std::string exec_name = get_executor_name(executor);

    // Allocate our data
    using vector_type = std::vector<STREAM_TYPE>;

    vector_type a(size);
    vector_type b(size);
    vector_type c(size);

    // Initialize arrays
    pika::fill(policy, a.begin(), a.end(), 1.0);
    pika::fill(policy, b.begin(), b.end(), 2.0);
    pika::fill(policy, c.begin(), c.end(), 0.0);

    ///////////////////////////////////////////////////////////////////////////
    // Warmup loop
    double scalar = 3.0;
    for (std::size_t iteration = 0; iteration != warmup_iterations; ++iteration)
    {
        // Copy
        pika::copy(policy, a.begin(), a.end(), c.begin());

        // Scale
        pika::transform(policy, c.begin(), c.end(), b.begin(),
            multiply_step<STREAM_TYPE>(scalar));

        // Add
        pika::ranges::transform(policy, a.begin(), a.end(), b.begin(), b.end(),
            c.begin(), add_step<STREAM_TYPE>());

        // Triad
        pika::ranges::transform(policy, b.begin(), b.end(), c.begin(), c.end(),
            a.begin(), triad_step<STREAM_TYPE>(scalar));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Reinitialize arrays (if needed)
    pika::fill(policy, a.begin(), a.end(), 1.0);
    pika::fill(policy, b.begin(), b.end(), 2.0);
    pika::fill(policy, c.begin(), c.end(), 0.0);

    // Copy
    pika::util::perftests_report("stream benchmark - Copy", exec_name,
        iterations,
        [&]() { pika::copy(policy, a.begin(), a.end(), c.begin()); });
    // Scale
    pika::util::perftests_report(
        "Stream benchmark - Scale", exec_name, iterations, [&]() {
            pika::transform(policy, c.begin(), c.end(), b.begin(),
                multiply_step<STREAM_TYPE>(scalar));
        });
    // Add
    pika::util::perftests_report(
        "Stream benchmark - Add", exec_name, iterations, [&]() {
            pika::ranges::transform(policy, a.begin(), a.end(), b.begin(),
                b.end(), c.begin(), add_step<STREAM_TYPE>());
        });
    // Triad
    pika::util::perftests_report(
        "Stream benchmark - Triad", exec_name, iterations, [&]() {
            pika::ranges::transform(policy, b.begin(), b.end(), c.begin(),
                c.end(), a.begin(), triad_step<STREAM_TYPE>(scalar));
        });

    // TODO: adapt the check result to work with the new version
    //// Check Results ...
    //check_results(iterations, a, b, c);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();
    std::size_t warmup_iterations = vm["warmup_iterations"].as<std::size_t>();
    std::size_t chunk_size = vm["chunk_size"].as<std::size_t>();
    std::size_t executor;
    header = vm.count("header") > 0;

    PIKA_UNUSED(chunk_size);

    if (vector_size < 1)
    {
        PIKA_THROW_EXCEPTION(pika::commandline_option_error, "pika_main",
            "Invalid vector size, must be at least 1");
    }

    if (iterations < 1)
    {
        PIKA_THROW_EXCEPTION(pika::commandline_option_error, "pika_main",
            "Invalid number of iterations given, must be at least 1");
    }

    {
        // Default parallel executor.
        executor = 0;
        using executor_type = pika::execution::parallel_executor;

        run_benchmark<>(warmup_iterations, iterations, vector_size,
            pika::execution::par.on(executor_type{}), executor);
    }

    {
        // Fork-join executor.
        executor = 1;
        using executor_type = pika::execution::experimental::fork_join_executor;

        run_benchmark<>(warmup_iterations, iterations, vector_size,
            pika::execution::par.on(executor_type{}), executor);
    }

    {
        // thread_pool_scheduler used through a scheduler_executor.
        executor = 2;
        using executor_type = pika::execution::experimental::scheduler_executor<
            pika::execution::experimental::thread_pool_scheduler>;

        run_benchmark<>(warmup_iterations, iterations, vector_size,
            pika::execution::par.on(executor_type{}), executor);
    }

    pika::util::perftests_print_times();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    using namespace pika::program_options;

    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        (   "csv", "output results as csv")
        (   "header", "print header for csv results")
        (   "vector_size",
            pika::program_options::value<std::size_t>()->default_value(1024),
            "size of vector (default: 1024)")
        (   "iterations",
            pika::program_options::value<std::size_t>()->default_value(10),
            "number of iterations to repeat each test. (default: 10)")
        (   "warmup_iterations",
            pika::program_options::value<std::size_t>()->default_value(1),
            "number of warmup iterations to perform before timing. (default: 1)")
        (   "chunk_size",
             pika::program_options::value<std::size_t>()->default_value(0),
            "size of vector (default: 1024)")
        ;
    // clang-format on

    // parse command line here to extract the necessary settings for pika
    parsed_options opts = command_line_parser(argc, argv)
                              .allow_unregistered()
                              .options(cmdline)
                              .style(command_line_style::unix_style)
                              .run();

    variables_map vm;
    store(opts, vm);

    std::vector<std::string> cfg = {
        "pika.numa_sensitive=2"    // no-cross NUMA stealing
    };

    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
