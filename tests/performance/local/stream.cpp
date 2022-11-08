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

#include <pika/algorithm.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/thread.hpp>
#include <pika/type_support/unused.hpp>
#include <pika/version.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

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
pika::threads::detail::topology& retrieve_topology()
{
    static pika::threads::detail::topology& topo =
        pika::threads::detail::create_topology();
    return topo;
}

///////////////////////////////////////////////////////////////////////////////
int checktick()
{
    static const std::size_t M = 20;
    using namespace std::chrono;
    time_point<high_resolution_clock> timesfound[M];

    // Collect a sequence of M unique time values from the system.
    for (std::size_t i = 0; i < M; i++)
    {
        auto const t1 = high_resolution_clock::now();
        time_point<high_resolution_clock> t2;
        while (
            duration<double>((t2 = high_resolution_clock::now()) - t1).count() <
            1.0E-6)
            ;
        timesfound[i] = t2;
    }

    // Determine the minimum difference between these M values.
    // This result will be our estimate (in microseconds) for the
    // clock granularity.
    int minDelta = 1000000;
    for (std::size_t i = 1; i < M; i++)
    {
        int Delta =
            duration_cast<milliseconds>(timesfound[i] - timesfound[i - 1])
                .count();
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
        fmt::print(std::cout, "WEIRD: sizeof(STREAM_TYPE) = {}\n",
            sizeof(STREAM_TYPE));
        epsilon = 1.e-6;
    }

    err = 0;
    if (std::abs(aAvgErr / aj) > epsilon)
    {
        err++;
        fmt::print(std::cout,
            "Failed Validation on array a[], AvgRelAbsErr > epsilon ({})\n",
            epsilon);
        fmt::print(std::cout,
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
                    fmt::print(std::cout,
                        "         array a: index: {}, expected: {}, "
                        "observed: {}, relative error: {}\n",
                        (unsigned long) j, aj, a[j],
                        (double) std::abs((aj - a[j]) / aAvgErr));
                }
#endif
            }
        }
        fmt::print(
            std::cout, "     For array a[], {} errors were found.\n", ierr);
    }
    if (std::abs(bAvgErr / bj) > epsilon)
    {
        err++;
        fmt::print(std::cout,
            "Failed Validation on array b[], AvgRelAbsErr > epsilon ({})\n",
            epsilon);
        fmt::print(std::cout,
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}\n", bj,
            bAvgErr, std::abs(bAvgErr) / bj);
        fmt::print(std::cout, "     AvgRelAbsErr > Epsilon ({})\n", epsilon);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(b[j] / bj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    fmt::print(std::cout,
                        "         array b: index: {}, expected: {}, "
                        "observed: {}, relative error: {}\n",
                        (unsigned long) j, bj, b[j],
                        (double) std::abs((bj - b[j]) / bAvgErr));
                }
#endif
            }
        }
        fmt::print(
            std::cout, "     For array b[], {} errors were found.\n", ierr);
    }
    if (std::abs(cAvgErr / cj) > epsilon)
    {
        err++;
        fmt::print(std::cout,
            "Failed Validation on array c[], AvgRelAbsErr > epsilon ({})\n",
            epsilon);
        fmt::print(std::cout,
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}\n", cj,
            cAvgErr, std::abs(cAvgErr) / cj);
        fmt::print(std::cout, "     AvgRelAbsErr > Epsilon ({})\n", epsilon);
        ierr = 0;
        for (std::size_t j = 0; j < a.size(); j++)
        {
            if (std::abs(c[j] / cj - 1.0) > epsilon)
            {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10)
                {
                    fmt::print(std::cout,
                        "         array c: index: {}, expected: {}, "
                        "observed: {}, relative error: {}\n",
                        (unsigned long) j, cj, c[j],
                        (double) std::abs((cj - c[j]) / cAvgErr));
                }
#endif
            }
        }
        fmt::print(
            std::cout, "     For array c[], {} errors were found.\n", ierr);
    }
    if (err == 0)
    {
        if (!csv)
        {
            fmt::print(std::cout,
                "Solution Validates: avg error less than {} on all three "
                "arrays\n",
                epsilon);
        }
    }
#ifdef VERBOSE
    fmt::print(std::cout, "Results Validation Verbose Results:\n");
    fmt::print(
        std::cout, "    Expected a(1), b(1), c(1): {} {} {}\n", aj, bj, cj);
    fmt::print(std::cout, "    Observed a(1), b(1), c(1): {} {} {}\n", a[1],
        b[1], c[1]);
    fmt::print(std::cout, "    Rel Errors on a, b, c:     {} {} {}\n",
        (double) std::abs(aAvgErr / aj), (double) std::abs(bAvgErr / bj),
        (double) std::abs(cAvgErr / cj));
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct multiply_step
{
    multiply_step(T factor)
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
    triad_step(T factor)
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
std::vector<std::vector<std::chrono::duration<double>>>
run_benchmark(std::size_t warmup_iterations, std::size_t iterations,
    std::size_t size, Policy&& policy)
{
    // Allocate our data
    using vector_type = std::vector<STREAM_TYPE>;

    vector_type a(size);
    vector_type b(size);
    vector_type c(size);

    // Initialize arrays
    pika::fill(policy, a.begin(), a.end(), 1.0);
    pika::fill(policy, b.begin(), b.end(), 2.0);
    pika::fill(policy, c.begin(), c.end(), 0.0);

    // Check clock ticks ...
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    pika::transform(
        policy, a.begin(), a.end(), a.begin(), multiply_step<STREAM_TYPE>(2.0));
    auto dur =
        duration_cast<microseconds>(high_resolution_clock::now() - start);

    // Get initial value for system clock.
    int quantum = checktick();
    if (quantum >= 1)
    {
        if (!csv)
        {
            std::cout << "Your clock granularity/precision appears to be "
                      << quantum << " microseconds.\n";
        }
    }
    else
    {
        if (!csv)
        {
            std::cout << "Your clock granularity appears to be less than one "
                         "microsecond.\n";
        }
        quantum = 1;
    }

    if (!csv)
    {
        // clang-format off
        std::cout << "Each test below will take on the order of "
                  << dur.count() << " microseconds.\n"
                  << "   (= " << (int) (dur.count() / quantum) << " clock ticks)\n"
                  << "Increase the size of the arrays if this shows that\n"
                  << "you are not getting at least 20 clock ticks per test.\n"
                  << "-------------------------------------------------------------\n";
        // clang-format on
    }

    if (!csv)
    {
        // clang-format off
        std::cout << "WARNING -- The above is only a rough guideline.\n"
                  << "For best results, please be sure you know the\n"
                  << "precision of your system timer.\n"
                  << "-------------------------------------------------------------\n";
        // clang-format on
    }

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

    using namespace std::chrono;
    time_point<high_resolution_clock> start_tmp;
    ///////////////////////////////////////////////////////////////////////////
    // Main timing loop
    std::vector<std::vector<duration<double>>> timing(
        4, std::vector<duration<double>>(iterations));
    for (std::size_t iteration = 0; iteration != iterations; ++iteration)
    {
        // Copy
        start_tmp = high_resolution_clock::now();
        pika::copy(policy, a.begin(), a.end(), c.begin());
        timing[0][iteration] = high_resolution_clock::now() - start_tmp;

        // Scale
        start_tmp = high_resolution_clock::now();
        pika::transform(policy, c.begin(), c.end(), b.begin(),
            multiply_step<STREAM_TYPE>(scalar));
        timing[1][iteration] =
            duration_cast<seconds>(high_resolution_clock::now() - start_tmp);

        // Add
        start_tmp = high_resolution_clock::now();
        pika::ranges::transform(policy, a.begin(), a.end(), b.begin(), b.end(),
            c.begin(), add_step<STREAM_TYPE>());
        timing[2][iteration] =
            duration_cast<seconds>(high_resolution_clock::now() - start_tmp);

        // Triad
        start_tmp = high_resolution_clock::now();
        pika::ranges::transform(policy, b.begin(), b.end(), c.begin(), c.end(),
            a.begin(), triad_step<STREAM_TYPE>(scalar));
        timing[3][iteration] =
            duration_cast<seconds>(high_resolution_clock::now() - start_tmp);
    }

    // Check Results ...
    check_results(iterations, a, b, c);

    if (!csv)
    {
        // clang-format off
        std::cout << "-------------------------------------------------------------\n";
        // clang-format on
    }

    return timing;
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();
    std::size_t offset = vm["offset"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();
    std::size_t warmup_iterations = vm["warmup_iterations"].as<std::size_t>();
    std::size_t chunk_size = vm["chunk_size"].as<std::size_t>();
    std::size_t executor = vm["executor"].as<std::size_t>();
    csv = vm.count("csv") > 0;
    header = vm.count("header") > 0;

    PIKA_UNUSED(chunk_size);

    std::string chunker = vm["chunker"].as<std::string>();

    if (vector_size < 1)
    {
        PIKA_THROW_EXCEPTION(pika::error::commandline_option_error, "pika_main",
            "Invalid vector size, must be at least 1");
    }

    if (iterations < 1)
    {
        PIKA_THROW_EXCEPTION(pika::error::commandline_option_error, "pika_main",
            "Invalid number of iterations given, must be at least 1");
    }

    // clang-format off
    if (!csv)
    {
        std::cout
            << "-------------------------------------------------------------\n"
            << "Modified STREAM benchmark based on\npika version: "
                << pika::build_string() << "\n"
            << "-------------------------------------------------------------\n"
            << "This system uses " << sizeof(STREAM_TYPE)
                << " bytes per array element.\n"
            << "-------------------------------------------------------------\n"
            << "Array size = " << vector_size << " (elements), "
               "Offset = " << offset << " (elements)\n"
            << "Memory per array = "
                << sizeof(STREAM_TYPE) * (vector_size / 1024. / 1024.) << " MiB "
            << "(= "
                <<  sizeof(STREAM_TYPE) * (vector_size / 1024. / 1024. / 1024.)
                << " GiB).\n"
            << "Each kernel will be executed " << iterations << " times.\n"
            << " The *best* time for each kernel (excluding the first iteration)\n"
            << " will be used to compute the reported bandwidth.\n"
            << "-------------------------------------------------------------\n"
            << "Number of Threads requested = "
                << pika::get_os_thread_count() << "\n"
            << "Chunking policy requested: " << chunker << "\n"
            << "Executor requested: " << executor << "\n"
            << "-------------------------------------------------------------\n"
            ;
    }
    // clang-format on

    using namespace std::chrono;
    auto start_total = high_resolution_clock::now();
    std::vector<std::vector<duration<double>>> timing;

    if (executor == 0)
    {
        // Default parallel policy with serial allocator.
        timing = run_benchmark<>(
            warmup_iterations, iterations, vector_size, pika::execution::par);
    }
    else if (executor == 1)
    {
        // Fork-join executor.
        using executor_type = pika::execution::experimental::fork_join_executor;

        executor_type exec;
        auto policy = pika::execution::par.on(exec);
        timing = run_benchmark<>(
            warmup_iterations, iterations, vector_size, std::move(policy));
    }
    else if (executor == 2)
    {
        // thread_pool_scheduler used through a scheduler_executor.
        using executor_type = pika::execution::experimental::scheduler_executor<
            pika::execution::experimental::thread_pool_scheduler>;

        executor_type exec;
        auto policy = pika::execution::par.on(exec);
        timing = run_benchmark<>(
            warmup_iterations, iterations, vector_size, std::move(policy));
    }
    else
    {
        PIKA_THROW_EXCEPTION(pika::error::commandline_option_error, "pika_main",
            "Invalid executor id given (0-4 allowed");
    }
    duration<double> dur_total = high_resolution_clock::now() - start_total;

    /* --- SUMMARY --- */
    // clang-format off
    std::size_t const num_stream_tests = 4;
    const char *label[num_stream_tests] = {
        "Copy:      ",
        "Scale:     ",
        "Add:       ",
        "Triad:     "
    };
    // clang-format on

    const double bytes[num_stream_tests] = {
        2 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size),
        2 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size),
        3 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size),
        3 * sizeof(STREAM_TYPE) * static_cast<double>(vector_size)};

    std::vector<double> avgtime(num_stream_tests, 0.0);
    std::vector<double> mintime(
        num_stream_tests, (std::numeric_limits<double>::max)());
    std::vector<double> maxtime(num_stream_tests, 0.0);

    for (std::size_t j = 0; j < num_stream_tests; j++)
    {
        avgtime[j] = timing[j][0].count();
        mintime[j] = timing[j][0].count();
        maxtime[j] = timing[j][0].count();
    }

    for (std::size_t iteration = 1; iteration != iterations; ++iteration)
    {
        for (std::size_t j = 0; j < num_stream_tests; j++)
        {
            avgtime[j] += timing[j][iteration].count();
            mintime[j] = (std::min)(mintime[j], timing[j][iteration].count());
            maxtime[j] = (std::max)(maxtime[j], timing[j][iteration].count());
        }
    }

    if (csv)
    {
        if (header)
        {
            fmt::print(std::cout,
                "executor,threads,vector_size,copy_bytes,copy_bw,copy_avg,copy_"
                "min,copy_max,scale_bytes,scale_bw,scale_avg,scale_min,scale_"
                "max,add_bytes,add_bw,add_avg,add_min,add_max,triad_bytes,"
                "triad_bw,triad_avg,triad_min,triad_max\n");
        }
        std::size_t const num_executors = 5;
        const char* executors[num_executors] = {"parallel-serial", "block",
            "parallel-parallel", "fork_join_executor", "scheduler_executor"};
        fmt::print(std::cout, "{},{},{},", executors[executor],
            pika::get_os_thread_count(), vector_size);
    }
    else
    {
        fmt::print(std::cout,
            "Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    }

    for (std::size_t j = 0; j < num_stream_tests; j++)
    {
        avgtime[j] = avgtime[j] / (double) (iterations);

        if (csv)
        {
            fmt::print(std::cout, "{:.0},{:.2},{:.9},{:.9},{:.9}{}", bytes[j],
                1.0E-06 * bytes[j] / mintime[j], avgtime[j], mintime[j],
                maxtime[j], j < num_stream_tests - 1 ? "," : "\n");
        }
        else
        {
            fmt::print(std::cout, "{}{:12.1}  {:11.6}  {:11.6}  {:11.6}\n",
                label[j], 1.0E-06 * bytes[j] / mintime[j], avgtime[j],
                mintime[j], maxtime[j]);
        }
    }

    if (!csv)
    {
        std::cout << "\nTotal time (s): " << dur_total.count()
                  << " (per iteration: " << dur_total.count() / iterations
                  << ")\n";
    }

    if (!csv)
    {
        // clang-format off
        std::cout << "-------------------------------------------------------------\n";
        // clang-format on
    }

    return pika::finalize();
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
        (   "offset",
            pika::program_options::value<std::size_t>()->default_value(0),
            "offset (default: 0)")
        (   "iterations",
            pika::program_options::value<std::size_t>()->default_value(10),
            "number of iterations to repeat each test. (default: 10)")
        (   "warmup_iterations",
            pika::program_options::value<std::size_t>()->default_value(1),
            "number of warmup iterations to perform before timing. (default: 1)")
        (   "chunker",
            pika::program_options::value<std::string>()->default_value("default"),
            "Which chunker to use for the parallel algorithms. "
            "possible values: dynamic, auto, guided. (default: default)")
        (   "chunk_size",
             pika::program_options::value<std::size_t>()->default_value(0),
            "size of vector (default: 1024)")
        (   "executor",
            pika::program_options::value<std::size_t>()->default_value(2),
            "executor to use (0-2) (default: 0, parallel_executor)")
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

    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return pika::init(pika_main, argc, argv, init_args);
}
