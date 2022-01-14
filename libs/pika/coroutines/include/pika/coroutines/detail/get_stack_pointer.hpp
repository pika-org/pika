//  Copyright (c) 2013-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/compiler_specific.hpp>

#if defined(PIKA_WINDOWS)
#define PIKA_HAVE_THREADS_GET_STACK_POINTER
#else
#if defined(PIKA_GCC_VERSION)
#define PIKA_HAVE_THREADS_GET_STACK_POINTER
#else
#if defined(__x86_64__) || defined(__amd64) || defined(__i386__) ||            \
    defined(__i486__) || defined(__i586__) || defined(__i686__) ||             \
    defined(__powerpc__) || defined(__arm__)
#define PIKA_HAVE_THREADS_GET_STACK_POINTER
#endif
#endif

#include <cstddef>
#include <limits>

namespace pika { namespace threads { namespace coroutines { namespace detail {

    inline std::size_t get_stack_ptr()
    {
#if defined(PIKA_GCC_VERSION)
        return std::size_t(__builtin_frame_address(0));
#else
        std::size_t stack_ptr = (std::numeric_limits<std::size_t>::max)();
#if defined(__x86_64__) || defined(__amd64)
        asm("movq %%rsp, %0" : "=r"(stack_ptr));
#elif defined(__i386__) || defined(__i486__) || defined(__i586__) ||           \
    defined(__i686__)
        asm("movl %%esp, %0" : "=r"(stack_ptr));
#elif defined(__powerpc__)
        void* stack_ptr_p = &stack_ptr;
        asm("stw %%r1, 0(%0)" : "=&r"(stack_ptr_p));
#elif defined(__arm__)
        asm("mov %0, sp" : "=r"(stack_ptr));
#endif
        return stack_ptr;
#endif
    }
}}}}    // namespace pika::threads::coroutines::detail
#endif
