//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !(defined(__x86_64__) || defined(__amd64__))
#error This file is for x86 CPUs only.
#endif

#if !defined(__GNUC__)
#error This file requires compilation with gcc.
#endif

//     RDI is &from.sp
//     RSI is to.sp
//
//     This is the simplest version of swapcontext
//     It saves registers on the old stack, saves the old stack pointer,
//     load the new stack pointer, pop registers from the new stack
//     and returns to new caller.
//
//     RDI is set to be the parameter for the function to be called.
//     The first time RDI is the first parameter of the trampoline.
//     Otherwise it is simply discarded.
//
//     NOTE: This function should work on any IA64 CPU.
//     NOTE: The biggest penalty is the last jump that
//           will be always mis-predicted (~50 cycles on P4).
//
//     We try to make its address available as soon as possible
//     to try to reduce the penalty. Doing a return instead of a
//
//        'add $8, %esp'
//        'jmp *%ecx'
//
//     really kills performance.
//
//     NOTE: popl is slightly better than mov+add to pop registers
//           so is pushl rather than mov+sub.

#if defined(__APPLE__)
#define PIKA_COROUTINE_TYPE_DIRECTIVE(name)
#else
#define PIKA_COROUTINE_TYPE_DIRECTIVE(name) ".type " #name ", @function\n\t"
#endif

// Note: .align 4 below means alignment at 2^4 boundary (16 bytes

#define PIKA_COROUTINE_SWAPCONTEXT(name)                                       \
    asm (                                                                     \
        ".text \n\t"                                                          \
        ".align 4\n"                                                          \
        ".globl " #name "\n\t"                                                \
        PIKA_COROUTINE_TYPE_DIRECTIVE(name)                                    \
    #name ":\n\t"                                                             \
        "movq  64(%rsi), %rcx\n\t"                                            \
        "pushq %rbp\n\t"                                                      \
        "pushq %rbx\n\t"                                                      \
        "pushq %rax\n\t"                                                      \
        "pushq %rdx\n\t"                                                      \
        "pushq %r12\n\t"                                                      \
        "pushq %r13\n\t"                                                      \
        "pushq %r14\n\t"                                                      \
        "pushq %r15\n\t"                                                      \
        "movq  %rsp, (%rdi)\n\t"                                              \
        "movq  %rsi, %rsp\n\t"                                                \
        "popq  %r15\n\t"                                                      \
        "popq  %r14\n\t"                                                      \
        "popq  %r13\n\t"                                                      \
        "popq  %r12\n\t"                                                      \
        "popq  %rdx\n\t"                                                      \
        "popq  %rax\n\t"                                                      \
        "popq  %rbx\n\t"                                                      \
        "popq  %rbp\n\t"                                                      \
        "movq 80(%rsi), %rdi\n\t"                                             \
        "add   $8, %rsp\n\t"                                                  \
        "jmp   *%rcx\n\t"                                                     \
        "ud2\n\t"                                                             \
    )                                                                         \
/**/

PIKA_COROUTINE_SWAPCONTEXT(swapcontext_stack);
PIKA_COROUTINE_SWAPCONTEXT(swapcontext_stack2);

#undef PIKA_COROUTINE_SWAPCONTEXT
#undef PIKA_COROUTINE_TYPE_DIRECTIVE

