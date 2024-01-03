//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_pool.hpp>
#include <pika/async_cuda/then_with_stream.hpp>
#include <pika/async_cuda_base/cuda_stream.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/execution/algorithms/bulk.hpp>
#include <pika/execution/algorithms/execute.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/iterator_support/range.hpp>

#include <whip.hpp>

#include <type_traits>

namespace pika::cuda::experimental {
    namespace detail {
        // This silences a bogus warning from nvcc about no return from a
        // non-void function.
        template <typename Shape>
        auto shape_size_impl(std::true_type, Shape&& shape)
        {
            return shape;
        }

        template <typename Shape>
        auto shape_size_impl(std::false_type, Shape&& shape)
        {
            return pika::util::size(PIKA_FORWARD(Shape, shape));
        }

        template <typename Shape>
        auto shape_size(Shape&& shape)
        {
            return shape_size_impl(
                std::is_integral<std::decay_t<Shape>>{}, PIKA_FORWARD(Shape, shape));
        }

#if defined(PIKA_COMPUTE_CODE)
        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename Shape>
        PIKA_DEVICE auto shape_dereference_impl(std::true_type, Shape&&, int i)
        {
            return static_cast<std::decay_t<Shape>>(i);
        }

        PIKA_NVCC_PRAGMA_HD_WARNING_DISABLE
        template <typename Shape>
        PIKA_DEVICE auto shape_dereference_impl(std::false_type, Shape&& shape, int i)
        {
            return pika::util::begin(shape)[i];
        }

        template <typename Shape>
        PIKA_DEVICE auto shape_dereference(Shape&& shape, int i)
        {
            return shape_dereference_impl(
                std::is_integral<std::decay_t<Shape>>{}, PIKA_FORWARD(Shape, shape), i);
        }

        template <typename F, typename Shape, typename Size, typename... Ts>
        __global__ void bulk_function_kernel_integral(F f, Shape shape, Size n, Ts... ts)
        {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < static_cast<int>(n)) { f(shape_dereference(shape, i), ts...); }
        }
#endif

        template <typename F, typename Shape>
        void launch_bulk_function(F f, Shape shape, whip::stream_t stream)
        {
#if defined(PIKA_COMPUTE_CODE)
            auto n = shape_size(shape);
            if (n > 0)
            {
                constexpr int block_dim = 256;
                int grid_dim = (n + block_dim - 1) / block_dim;

                bulk_function_kernel_integral<<<block_dim, grid_dim, 0, stream>>>(f, shape, n);
                whip::check_last_error();
            }
#else
            PIKA_UNUSED(f);
            PIKA_UNUSED(shape);
            PIKA_UNUSED(stream);
#endif
        }

        template <typename F, typename Shape, typename T>
        void launch_bulk_function(F f, Shape shape, T t, whip::stream_t stream)
        {
#if defined(PIKA_COMPUTE_CODE)
            auto n = shape_size(shape);
            if (n > 0)
            {
                constexpr int block_dim = 256;
                int grid_dim = (n + block_dim - 1) / block_dim;

                bulk_function_kernel_integral<<<block_dim, grid_dim, 0, stream>>>(f, shape, n, t);
                whip::check_last_error();
            }
#else
            PIKA_UNUSED(f);
            PIKA_UNUSED(shape);
            PIKA_UNUSED(t);
            PIKA_UNUSED(stream);
#endif
        }

        template <typename Shape, typename F>
        struct bulk_launcher
        {
            std::decay_t<Shape> shape;
            std::decay_t<F> f;

            // TODO: This should pass through the additional arguments.
            // Currently only works with zero or one arguments with a plain call
            // operator like here.
            void operator()(whip::stream_t stream) const { launch_bulk_function(f, shape, stream); }

            template <typename T>
            std::decay_t<T> operator()(T&& t, whip::stream_t stream) const
            {
                launch_bulk_function(f, shape, t, stream);
                return PIKA_MOVE(t);
            }
        };
    }    // namespace detail

    /// Execute a function in bulk on a CUDA device.
    template <typename Sender, typename Shape, typename F>
    decltype(auto) tag_invoke(pika::execution::experimental::bulk_t, cuda_scheduler,
        Sender&& sender, Shape&& shape, F&& f)
    {
        return then_with_stream(PIKA_FORWARD(Sender, sender),
            detail::bulk_launcher<Shape, F>{PIKA_FORWARD(Shape, shape), PIKA_FORWARD(F, f)});
    }
}    // namespace pika::cuda::experimental
