//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/async_cuda/cuda_pool.hpp>
#include <pika/async_cuda/cuda_scheduler.hpp>

namespace pika::cuda::experimental {
    cuda_scheduler::cuda_scheduler(cuda_pool pool)
      : pool(PIKA_MOVE(pool))
      , priority(pika::execution::thread_priority::default_)
    {
    }

    cuda_pool const& cuda_scheduler::get_pool() const noexcept { return pool; }

    cuda_stream const& cuda_scheduler::get_next_stream() { return pool.get_next_stream(priority); }

    locked_cublas_handle cuda_scheduler::get_cublas_handle(
        cuda_stream const& stream, cublasPointerMode_t pointer_mode)
    {
        return pool.get_cublas_handle(stream, pointer_mode);
    }

    locked_cusolver_handle cuda_scheduler::get_cusolver_handle(cuda_stream const& stream)
    {
        return pool.get_cusolver_handle(stream);
    }

    namespace detail {
        cuda_scheduler_sender::cuda_scheduler_sender(cuda_scheduler scheduler)
          : scheduler(PIKA_MOVE(scheduler))
        {
        }
    }    // namespace detail
}    // namespace pika::cuda::experimental
