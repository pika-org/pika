
//  Copyright (c) 2011-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "jacobi_nonuniform.hpp"

#include <pika/execution.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/timing/high_resolution_timer.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

namespace jacobi_smp {

    void jacobi_kernel_wrap(range const& r, crs_matrix<double> const& A, std::vector<double>& x_dst,
        std::vector<double> const& x_src, std::vector<double> const& b)
    {
        for (std::size_t row = r.begin(); row < r.end(); ++row)
        {
            jacobi_kernel_nonuniform(A, x_dst, x_src, b, row);
        }
    }

    void jacobi(crs_matrix<double> const& A, std::vector<double> const& b, std::size_t iterations,
        std::size_t block_size)
    {
        using vector_type = std::vector<double>;

        std::shared_ptr<vector_type> dst(new vector_type(b));
        std::shared_ptr<vector_type> src(new vector_type(b));

        std::vector<range> block_ranges;
        // pre-computing ranges for the different blocks
        for (std::size_t i = 0; i < dst->size(); i += block_size)
        {
            block_ranges.push_back(range(i, std::min<std::size_t>(dst->size(), i + block_size)));
        }

        // pre-computing dependencies
        std::vector<std::vector<std::size_t>> dependencies(block_ranges.size());
        for (std::size_t b = 0; b < block_ranges.size(); ++b)
        {
            for (std::size_t i = block_ranges[b].begin(); i < block_ranges[b].end(); ++i)
            {
                std::size_t begin = A.row_begin(i);
                std::size_t end = A.row_end(i);

                for (std::size_t ii = begin; ii < end; ++ii)
                {
                    std::size_t idx = A.indices[ii];
                    for (std::size_t j = 0; j < block_ranges.size(); ++j)
                    {
                        if (block_ranges[j].begin() <= idx && idx < block_ranges[j].end())
                        {
                            if (std::find(dependencies[b].begin(), dependencies[b].end(), j) ==
                                dependencies[b].end())
                            {
                                dependencies[b].push_back(j);
                            }
                            break;
                        }
                    }
                }
            }
        }

        using sender_vector = std::vector<ex::any_sender<>>;
        std::shared_ptr<sender_vector> deps_dst(new sender_vector(dependencies.size(), ex::just()));
        std::shared_ptr<sender_vector> deps_src(new sender_vector(dependencies.size(), ex::just()));

        pika::chrono::detail::high_resolution_timer t;
        for (std::size_t iter = 0; iter < iterations; ++iter)
        {
            for (std::size_t block = 0; block < block_ranges.size(); ++block)
            {
                std::vector<std::size_t> const& deps(dependencies[block]);
                sender_vector trigger;
                trigger.reserve(deps.size());
                for (std::size_t dep : deps) { trigger.push_back((*deps_src)[dep]); }

                (*deps_dst)[block] = ex::when_all_vector(std::move(trigger)) |
                    ex::continues_on(ex::thread_pool_scheduler{}) |
                    ex::then(pika::util::detail::bind_front(jacobi_kernel_wrap, block_ranges[block],
                        std::cref(A), std::ref(*dst), std::cref(*src), std::cref(b))) |
                    ex::split();
            }
            std::swap(dst, src);
            std::swap(deps_dst, deps_src);
        }

        tt::sync_wait(ex::when_all_vector(std::move(*deps_dst)));
        tt::sync_wait(ex::when_all_vector(std::move(*deps_src)));

        double time_elapsed = t.elapsed<std::chrono::seconds>();
        std::cout << dst->size() << " " << ((double(dst->size() * iterations) / 1e6) / time_elapsed)
                  << " MLUPS/s\n"
                  << std::flush;
    }
}    // namespace jacobi_smp
