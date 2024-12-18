//  Copyright (c) 2011-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/timing/high_resolution_timer.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "jacobi.hpp"

namespace jacobi_smp {
    void jacobi(
        std::size_t n, std::size_t iterations, std::size_t, std::string const& output_filename)
    {
        using vector = std::vector<double>;

        std::shared_ptr<vector> grid_new(new vector(n * n, 1));
        std::shared_ptr<vector> grid_old(new vector(n * n, 1));

        pika::chrono::detail::high_resolution_timer t;
        for (std::size_t i = 0; i < iterations; ++i)
        {
            // MSVC is unhappy if the OMP loop variable is unsigned
#pragma omp parallel for schedule(JACOBI_SMP_OMP_SCHEDULE)
            for (std::int64_t y = 1; y < std::int64_t(n - 1); ++y)
            {
                double* dst = &(*grid_new)[y * n];
                double const* src = &(*grid_new)[y * n];
                jacobi_kernel(dst, src, n);
            }
            std::swap(grid_new, grid_old);
        }

        report_timing(n, iterations, t.elapsed<std::chrono::seconds>());
        output_grid(output_filename, *grid_old, n);
    }
}    // namespace jacobi_smp
