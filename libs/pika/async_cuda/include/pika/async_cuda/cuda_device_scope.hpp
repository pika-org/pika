//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

namespace pika::cuda::experimental {
    /// RAII wrapper for setting and unsetting the current CUDA device.
    ///
    /// Stores the current device on construction, sets the given device, and
    /// resets the device on destruction.
    class [[nodiscard]] cuda_device_scope
    {
    private:
        int device;
        int old_device;

    public:
        PIKA_EXPORT explicit cuda_device_scope(int device = 0);
        PIKA_EXPORT ~cuda_device_scope();
        cuda_device_scope(cuda_device_scope&&) = delete;
        cuda_device_scope& operator=(cuda_device_scope&&) = delete;
        cuda_device_scope(cuda_device_scope const&) = delete;
        cuda_device_scope& operator=(cuda_device_scope const&) = delete;
    };
}    // namespace pika::cuda::experimental
