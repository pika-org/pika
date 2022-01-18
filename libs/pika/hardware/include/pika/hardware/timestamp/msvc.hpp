////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_WINDOWS)

#include <cstdint>

#include <intrin.h>
#include <windows.h>

#if defined(PIKA_HAVE_CUDA) && defined(PIKA_COMPUTE_CODE)
#include <pika/hardware/timestamp/cuda.hpp>
#endif

namespace pika { namespace util { namespace hardware {
    PIKA_HOST_DEVICE inline std::uint64_t timestamp()
    {
#if defined(PIKA_HAVE_CUDA) && defined(PIKA_COMPUTE_DEVICE_CODE)
        return timestamp_cuda();
#else
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return static_cast<std::uint64_t>(now.QuadPart);
#endif
    }
}}}    // namespace pika::util::hardware

#endif
