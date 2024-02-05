//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file provides functionality similar to CUDA's built-in
// cudaStreamAddCallback, with the difference that an event is recorded and an
// pika scheduler polls for the completion of the event. When the event is ready,
// a callback is called.

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda_base/cuda_stream.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <whip.hpp>

#include <string>

namespace pika::cuda::experimental::detail {
    using event_callback_function_type = pika::util::detail::unique_function<void(whip::error_t)>;

    PIKA_EXPORT void add_event_callback(event_callback_function_type&& f, whip::stream_t stream,
        pika::execution::thread_priority = pika::execution::thread_priority::default_);
    PIKA_EXPORT void add_event_callback(
        event_callback_function_type&& f, cuda_stream const& stream);

    PIKA_EXPORT void register_polling(pika::threads::detail::thread_pool_base& pool);
    PIKA_EXPORT void unregister_polling(pika::threads::detail::thread_pool_base& pool);
}    // namespace pika::cuda::experimental::detail
