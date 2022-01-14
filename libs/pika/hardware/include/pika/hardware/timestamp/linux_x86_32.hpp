////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>

#include <cstdint>

#if defined(PIKA_HAVE_CUDA) && defined(PIKA_COMPUTE_CODE)
#include <pika/hardware/timestamp/cuda.hpp>
#endif

namespace pika { namespace util { namespace hardware {

    // clang-format off
    PIKA_HOST_DEVICE inline std::uint64_t timestamp()
    {
#if defined(PIKA_HAVE_CUDA) && defined(PIKA_COMPUTE_DEVICE_CODE)
        return timestamp_cuda();
#else
        std::uint64_t r = 0;

        #if defined(PIKA_HAVE_RDTSCP)
            __asm__ __volatile__(
                "rdtscp ;\n"
                : "=A"(r)
                :
                : "%ecx");
        #elif defined(PIKA_HAVE_RDTSC)
            __asm__ __volatile__(
                "movl %%ebx, %%edi ;\n"
                "xorl %%eax, %%eax ;\n"
                "cpuid ;\n"
                "rdtsc ;\n"
                "movl %%edi, %%ebx ;\n"
                : "=A"(r)
                :
                : "%edi", "%ecx");
        #endif

        return r;
#endif
    }
    // clang-format on

}}}    // namespace pika::util::hardware
