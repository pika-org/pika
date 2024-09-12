
//  Copyright (c) 2011-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "jacobi.hpp"

#include <pika/execution.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/timing/high_resolution_timer.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

namespace jacobi_smp {

    void jacobi_kernel_wrap(range const& y_range, std::size_t n, std::vector<double>& dst,
        std::vector<double> const& src)
    {
        for (std::size_t y = y_range.begin(); y < y_range.end(); ++y)
        {
            double* dst_ptr = &dst[y * n];
            const double* src_ptr = &src[y * n];
            jacobi_kernel(dst_ptr, src_ptr, n);
        }
    }

    void jacobi(std::size_t n, std::size_t iterations, std::size_t block_size,
        std::string const& output_filename)
    {
        using vector = std::vector<double>;

        std::shared_ptr<vector> grid_new(new vector(n * n, 1));
        std::shared_ptr<vector> grid_old(new vector(n * n, 1));

        using deps_vector = std::vector<ex::any_sender<>>;

        std::size_t n_block = static_cast<std::size_t>(
            std::ceil(static_cast<double>(n) / static_cast<double>(block_size)));

        std::shared_ptr<deps_vector> deps_new(new deps_vector(n_block, ex::just()));
        std::shared_ptr<deps_vector> deps_old(new deps_vector(n_block, ex::just()));

        pika::chrono::detail::high_resolution_timer t;
        for (std::size_t i = 0; i < iterations; ++i)
        {
            for (std::size_t y = 1, j = 0; y < n - 1; y += block_size, ++j)
            {
                std::size_t y_end = (std::min)(y + block_size, n - 1);
                deps_vector trigger;
                trigger.reserve(3);
                trigger.push_back((*deps_old)[j]);
                if (j > 0) trigger.push_back((*deps_old)[j - 1]);
                if (j + 1 < n_block) trigger.push_back((*deps_old)[j + 1]);

                (*deps_new)[j] = ex::when_all_vector(std::move(trigger)) |
                    ex::continues_on(ex::thread_pool_scheduler{}) |
                    ex::then(pika::util::detail::bind_front(jacobi_kernel_wrap, range(y, y_end), n,
                        std::ref(*grid_new), std::cref(*grid_old))) |
                    ex::split();
            }

            std::swap(grid_new, grid_old);
            std::swap(deps_new, deps_old);
        }
        tt::sync_wait(ex::when_all_vector(std::move(*deps_new)));
        tt::sync_wait(ex::when_all_vector(std::move(*deps_old)));

        report_timing(n, iterations, t.elapsed<std::chrono::seconds>());
        output_grid(output_filename, *grid_old, n);
    }
}    // namespace jacobi_smp
