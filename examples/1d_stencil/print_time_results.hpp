//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

///////////////////////////////////////////////////////////////////////////////
void print_time_results(std::uint32_t num_localities,
    std::uint64_t num_os_threads, double elapsed_seconds, std::uint64_t nx,
    std::uint64_t np, std::uint64_t nt, bool header)
{
    if (header)
        std::cout << "Localities,OS_Threads,Execution_Time_sec,"
                     "Points_per_Partition,Partitions,Time_Steps\n"
                  << std::flush;

    std::string const locs_str = fmt::format("{},", num_localities);
    std::string const threads_str = fmt::format("{},", num_os_threads);
    std::string const nx_str = fmt::format("{},", nx);
    std::string const np_str = fmt::format("{},", np);
    std::string const nt_str = fmt::format("{} ", nt);

    fmt::print(std::cout, "{:6} {:6} {:.14g}, {:21} {:21} {:21}\n", locs_str,
        threads_str, elapsed_seconds, nx_str, np_str, nt_str);
}

///////////////////////////////////////////////////////////////////////////////
void print_time_results(std::uint64_t num_os_threads, double elapsed_seconds,
    std::uint64_t nx, std::uint64_t np, std::uint64_t nt, bool header)
{
    if (header)
        std::cout << "OS_Threads,Execution_Time_sec,"
                     "Points_per_Partition,Partitions,Time_Steps\n"
                  << std::flush;

    std::string const threads_str = fmt::format("{},", num_os_threads);
    std::string const nx_str = fmt::format("{},", nx);
    std::string const np_str = fmt::format("{},", np);
    std::string const nt_str = fmt::format("{} ", nt);

    fmt::print(std::cout, "{:21} {:.14g}, {:21} {:21} {:21}\n", threads_str,
        elapsed_seconds, nx_str, np_str, nt_str);
}

void print_time_results(std::uint64_t num_os_threads, double elapsed_seconds,
    std::uint64_t nx, std::uint64_t nt, bool header)
{
    if (header)
        std::cout << "OS_Threads,Execution_Time_sec,"
                     "Grid_Points,Time_Steps\n"
                  << std::flush;

    std::string const threads_str = fmt::format("{},", num_os_threads);
    std::string const nx_str = fmt::format("{},", nx);
    std::string const nt_str = fmt::format("{} ", nt);

    fmt::print(std::cout, "{:21} {:10.12f}, {:21} {:21}\n", threads_str,
        elapsed_seconds, nx_str, nt_str);
}
