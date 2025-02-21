//  Copyright (c) 2025 ETH ZUrich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/coroutines/detail/stackoverflow_detection.hpp>

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)

# include <cstddef>
# include <cstdlib>
# include <cstring>
# include <string_view>

# if defined(PIKA_HAVE_UNISTD_H)
#  include <unistd.h>
# endif

# include <signal.h>
# include <stdio.h>

# if !defined(PIKA_SEGV_STACK_SIZE)
#  define PIKA_SEGV_STACK_SIZE MINSIGSTKSZ + 4096
# endif

namespace pika::threads::coroutines::detail {
    // This is as bare-bones as possible because it might be called as a result of a stack overflow.
    // Note that it uses write directly instead of printf as it's safe to call from a signal handler
    // (see https://man7.org/linux/man-pages/man7/signal-safety.7.html). We format pointers manually
    // into fixed size stack-allocated char buffers, and avoid all dynamic memory allocation.
    static void sigsegv_handler([[maybe_unused]] int signum, siginfo_t* infoptr, void* ctxptr)
    {
        // printf is in theory not signal safe, but we can attempt to use it if write isn't
        // available, and the messages may be printed correctly anyway.
        constexpr auto write_helper = [](char const* msg, std::size_t size) {
# if defined(PIKA_HAVE_UNISTD_H)
            write(STDERR_FILENO, msg, size);
# else
            dprintf(STDERR_FILENO, "%s", msg);
# endif
        };

        constexpr std::string_view sigsegv_msg =
            "[pika] pika's SIGSEGV handler has been called; trying to print as much "
            "information as possible, but output may be incomplete. This may be a stack overflow. "
            "You can increase the stack sizes by modifying the configuration options "
            "PIKA_SMALL_STACK_SIZE, PIKA_MEDIUM_STACK_SIZE, PIKA_LARGE_STACK_SIZE, or "
            "PIKA_HUGE_STACK_SIZE.\n";
        write_helper(sigsegv_msg.data(), sigsegv_msg.size());

        ucontext_t* uc_ctx = static_cast<ucontext_t*>(ctxptr);
        char* sigsegv_ptr = static_cast<char*>(infoptr->si_addr);

        // https://www.gnu.org/software/libc/manual/html_node/Signal-Stack.html
        char* stk_ptr = static_cast<char*>(uc_ctx->uc_stack.ss_sp);

        std::ptrdiff_t addr_delta =
            (sigsegv_ptr > stk_ptr) ? (sigsegv_ptr - stk_ptr) : (stk_ptr - sigsegv_ptr);

        constexpr std::string_view segv_address_msg = "segv address:  0x";
        constexpr std::string_view stack_pointer_msg = "stack pointer: 0x";
        constexpr std::string_view diff_msg = "diff:          0x";

        // Format a pointer as a hex string. This is a local function since it assumes that buffer is
        // big enough. The buffer is not null-terminated, since we know how many bytes to write from
        // the buffer.
        constexpr std::size_t ptr_size = sizeof(void*);
        constexpr std::size_t ptr_msg_size = ptr_size * 2;
        constexpr auto format_ptr = [](char* buffer, void* p) {
            constexpr auto hex_digits = "0123456789abcdef";
            auto value = reinterpret_cast<std::uintptr_t>(p);
            for (std::size_t i = 0; i < ptr_msg_size; ++i)
            {
                buffer[ptr_msg_size - i - 1] = hex_digits[value & 0xf];
                value >>= 4;
            }
        };

        char ptr_buffer[ptr_msg_size + 1];
        ptr_buffer[ptr_msg_size] = '\0';
        std::string_view ptr_msg(ptr_buffer, ptr_msg_size);

        constexpr auto write_msg_ptr = [write_helper, format_ptr](
                                           std::string_view msg, char* ptr_buffer, void* p) {
            write_helper(msg.data(), msg.size());
            format_ptr(ptr_buffer, p);
            write_helper(ptr_buffer, ptr_msg_size);
            write_helper("\n", 1);
        };

        write_msg_ptr(segv_address_msg, ptr_buffer, sigsegv_ptr);
        write_msg_ptr(stack_pointer_msg, ptr_buffer, stk_ptr);
        write_msg_ptr(diff_msg, ptr_buffer, reinterpret_cast<void*>(addr_delta));

        // heuristic value 1 kilobyte
        constexpr std::size_t coroutine_stackoverflow_addr_epsilon = 1024;

        // check the stack addresses, if they're < coroutine_stackoverflow_addr_epsilon apart,
        // terminate program should filter segmentation faults caused by coroutine stack overflows
        // from 'genuine' stack overflows
        if (static_cast<size_t>(addr_delta) < coroutine_stackoverflow_addr_epsilon)
        {
            constexpr std::string_view maybe_stackoverflow_msg =
                "The diff is small; this is likely a stack overflow.\n";
            write_helper(maybe_stackoverflow_msg.data(), maybe_stackoverflow_msg.size());
        }

        std::abort();
    }

    struct stackoverflow_helper
    {
        stackoverflow_helper() noexcept
        {
            segv_stack.ss_sp = valloc(PIKA_SEGV_STACK_SIZE);
            segv_stack.ss_flags = 0;
            segv_stack.ss_size = PIKA_SEGV_STACK_SIZE;

            std::memset(&action, '\0', sizeof(action));
            action.sa_flags = SA_SIGINFO | SA_ONSTACK;
            action.sa_sigaction = &sigsegv_handler;

            sigaltstack(&segv_stack, nullptr);
            sigemptyset(&action.sa_mask);
            sigaddset(&action.sa_mask, SIGSEGV);
            sigaction(SIGSEGV, &action, nullptr);
        }

        stackoverflow_helper(stackoverflow_helper&&) = delete;
        stackoverflow_helper& operator=(stackoverflow_helper&&) = delete;

        stackoverflow_helper(stackoverflow_helper const&) = delete;
        stackoverflow_helper& operator=(stackoverflow_helper const&) = delete;

        ~stackoverflow_helper() noexcept { free(segv_stack.ss_sp); }

        stack_t segv_stack;
        struct sigaction action;
    };

    void set_sigsegv_handler()
    {
        // Set handler at most once per thread
        static thread_local stackoverflow_helper helper{};
    }
}    // namespace pika::threads::coroutines::detail
#else
namespace pika::threads::coroutines::detail {
    void set_sigsegv_handler() {}
}    // namespace pika::threads::coroutines::detail
#endif
