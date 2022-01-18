////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>

#include <time.h>

#include <cstdint>

#if defined(PIKA_HAVE_CUDA) && defined(PIKA_COMPUTE_CODE)
#include <pika/hardware/timestamp/cuda.hpp>
#endif

namespace pika { namespace util { namespace hardware {

    PIKA_HOST_DEVICE inline std::uint64_t timestamp()
    {
#if defined(PIKA_HAVE_CUDA) && defined(PIKA_COMPUTE_DEVICE_CODE)
        return timestamp_cuda();
#else
        struct timespec res;
        clock_gettime(CLOCK_MONOTONIC, &res);
        return 1000 * res.tv_sec + res.tv_nsec / 1000000;
#endif
    }

}}}    // namespace pika::util::hardware
