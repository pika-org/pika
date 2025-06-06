//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// NOTE (per http://lists.apple.com/archives/darwin-dev/2008/Jan/msg00232.html):
// > Why the bus error? What am I doing wrong?
// This is a known issue where getcontext(3) is writing past the end of the
// ucontext_t struct when _XOPEN_SOURCE is not defined (rdar://problem/5578699
// ). As a workaround, define _XOPEN_SOURCE before including ucontext.h.
#if defined(__APPLE__) && !defined(_XOPEN_SOURCE)
# define _XOPEN_SOURCE
// However, the above #define will only affect <ucontext.h> if it has not yet
// been #included by something else!
# if defined(_STRUCT_UCONTEXT)
#  error You must #include coroutine headers before anything that #includes <ucontext.h>
# endif
#endif

#include <pika/assert.hpp>
#include <pika/util/get_and_reset_value.hpp>

// include unist.d conditionally to check for POSIX version. Not all OSs have
// the unistd header...
#if defined(PIKA_HAVE_UNISTD_H)
# include <unistd.h>
#endif

#if defined(PIKA_HAVE_ADDRESS_SANITIZER)
# include <sanitizer/asan_interface.h>
#endif

#if defined(__FreeBSD__) ||                                                                        \
    (defined(_XOPEN_UNIX) && defined(_XOPEN_VERSION) && _XOPEN_VERSION >= 500) ||                  \
    defined(__bgq__) || defined(__powerpc__) || defined(__s390x__)

// OS X 10.4 -- despite passing the test above -- doesn't support
// swapcontext() et al. Use GNU Pth workalike functions.
# if defined(__APPLE__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 1050)

#  include <cerrno>
#  include <cstddef>
#  include <cstdint>
#  include <exception>
#  include <limits>

#  include "pth/pth.h"

namespace pika::threads::coroutines::detail::posix::pth {
    inline int check_(int rc)
    {
        // The makecontext() functions return zero for success, nonzero for
        // error. The Pth library returns TRUE for success, FALSE for error,
        // with errno set to the nonzero error in the latter case. Map the Pth
        // returns to ucontext style.
        return rc ? 0 : errno;
    }
}    // namespace pika::threads::coroutines::detail::posix::pth

#  define PIKA_COROUTINE_POSIX_IMPL "Pth implementation"
#  define PIKA_COROUTINE_DECLARE_CONTEXT(name) pth_uctx_t name
#  define PIKA_COROUTINE_CREATE_CONTEXT(ctx)                                                       \
      pika::threads::coroutines::detail::posix::pth::check_(pth_uctx_create(&(ctx)))
#  define PIKA_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto)               \
      /* const sigset_t* sigmask = nullptr: we don't expect per-context signal     \
   * masks */               \
      pika::threads::coroutines::detail::posix::pth::check_(pth_uctx_make(                         \
          *(ctx), static_cast<char*>(stack), (size), nullptr, (startfunc), (startarg), (exitto)))
#  define PIKA_COROUTINE_SWAP_CONTEXT(from, to)                                                    \
      pika::threads::coroutines::detail::posix::pth::check_(                                       \
          pth_uctx_switch(*(from), *(to))) #define PIKA_COROUTINE_DESTROY_CONTEXT(ctx)             \
          pika::threads::coroutines::detail::posix::pth::check_(pth_uctx_destroy(ctx))

# else                  // generic Posix platform (e.g. OS X >= 10.5)

/*
 * makecontext based context implementation. Should be available on all
 * SuSv2 compliant UNIX systems.
 * NOTE: this implementation is not
 * optimal as the makecontext API saves and restore the signal mask.
 * This requires a system call for every context switch that really kills
 * performance. Still is very portable and guaranteed to work.
 * NOTE2: makecontext and friends are declared obsolescent in SuSv3, but
 * it is unlikely that they will be removed any time soon.
 */
#  include <cstddef>    // ptrdiff_t
#  include <iomanip>
#  include <iostream>
#  include <ucontext.h>

namespace pika::threads::coroutines::detail::posix::ucontext {
    inline int make_context(::ucontext_t* ctx, void* stack, std::ptrdiff_t size,
        void (*startfunc)(void*), void* startarg, ::ucontext_t* exitto = nullptr)
    {
        int error = ::getcontext(ctx);
        if (error) return error;

        ctx->uc_stack.ss_sp = (char*) stack;
        ctx->uc_stack.ss_size = size;
        ctx->uc_link = exitto;

        using ctx_main = void (*)();
        // makecontext can't fail.
        ::makecontext(ctx, (ctx_main) (startfunc), 1, startarg);
        return 0;
    }
}    // namespace pika::threads::coroutines::detail::posix::ucontext

#  define PIKA_COROUTINE_POSIX_IMPL "ucontext implementation"
#  define PIKA_COROUTINE_DECLARE_CONTEXT(name) ::ucontext_t name
#  define PIKA_COROUTINE_CREATE_CONTEXT(ctx) /* nop */
#  define PIKA_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto)               \
      pika::threads::coroutines::detail::posix::ucontext::make_context(                            \
          ctx, stack, size, startfunc, startarg, exitto)
#  define PIKA_COROUTINE_SWAP_CONTEXT(pfrom, pto) ::swapcontext((pfrom), (pto))
#  define PIKA_COROUTINE_DESTROY_CONTEXT(ctx) /* nop */

# endif    // generic Posix platform

# include <pika/coroutines/detail/get_stack_pointer.hpp>
# include <pika/coroutines/detail/posix_utility.hpp>
# include <pika/coroutines/detail/swap_context.hpp>
# include <atomic>
# include <signal.h>    // SIGSTKSZ

namespace pika::threads::coroutines {
    namespace detail {
        // some platforms need special preparation of the main thread
        struct prepare_main_thread
        {
            constexpr prepare_main_thread() {}
        };
    }    // namespace detail

    namespace detail::posix {
        /// Posix implementation for the context_impl_base class.
        /// \note context_impl is not required to be consistent
        /// If not initialized it can only be swapped out, not in
        /// (at that point it will be initialized).
        class ucontext_context_impl_base : detail::context_impl_base
        {
        public:
            // on some platforms SIGSTKSZ resolves to a syscall, we can't make
            // this constexpr
            PIKA_EXPORT static std::ptrdiff_t default_stack_size;

            ucontext_context_impl_base() { PIKA_COROUTINE_CREATE_CONTEXT(m_ctx); }
            ~ucontext_context_impl_base() { PIKA_COROUTINE_DESTROY_CONTEXT(m_ctx); }

# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            void start_switch_fiber(void** fake_stack)
            {
                __sanitizer_start_switch_fiber(fake_stack, asan_stack_bottom, asan_stack_size);
            }
            void start_yield_fiber(void** fake_stack, ucontext_context_impl_base& caller)
            {
                __sanitizer_start_switch_fiber(
                    fake_stack, caller.asan_stack_bottom, caller.asan_stack_size);
            }
            void finish_yield_fiber(void* fake_stack)
            {
                __sanitizer_finish_switch_fiber(fake_stack, &asan_stack_bottom, &asan_stack_size);
            }
            void finish_switch_fiber(void* fake_stack, ucontext_context_impl_base& caller)
            {
                __sanitizer_finish_switch_fiber(
                    fake_stack, &caller.asan_stack_bottom, &caller.asan_stack_size);
            }
# endif

        private:
            /// Free function. Saves the current context in \p from
            /// and restores the context in \p to.
            friend void swap_context(ucontext_context_impl_base& from,
                ucontext_context_impl_base const& to, default_hint)
            {
                [[maybe_unused]] int error = PIKA_COROUTINE_SWAP_CONTEXT(&from.m_ctx, &to.m_ctx);
                PIKA_ASSERT(error == 0);
            }

        protected:
            PIKA_COROUTINE_DECLARE_CONTEXT(m_ctx);

# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
        public:
            void* asan_fake_stack = nullptr;
            void const* asan_stack_bottom = nullptr;
            std::size_t asan_stack_size = 0;
# endif
        };

        template <typename CoroutineImpl>
        class ucontext_context_impl : public ucontext_context_impl_base
        {
        public:
            PIKA_NON_COPYABLE(ucontext_context_impl);

        public:
            using context_impl_base = ucontext_context_impl_base;

            /// Create a context that on restore invokes Functor on
            ///  a new stack. The stack size can be optionally specified.
            explicit ucontext_context_impl(std::ptrdiff_t stack_size = -1)
              : m_stack_size(stack_size == -1 ? this->default_stack_size : stack_size)
              , m_stack(nullptr)
              , funp_(&trampoline<CoroutineImpl>)
            {
            }

            void init()
            {
                if (m_stack != nullptr) return;

                m_stack = alloc_stack(static_cast<std::size_t>(m_stack_size));
                if (m_stack == nullptr)
                {
                    throw std::runtime_error("could not allocate memory for stack");
                }

                [[maybe_unused]] int error = PIKA_COROUTINE_MAKE_CONTEXT(
                    &m_ctx, m_stack, m_stack_size, funp_, this, nullptr);

                PIKA_ASSERT(error == 0);

# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
                asan_stack_size = m_stack_size;
                asan_stack_bottom = const_cast<void const*>(m_stack);
# endif
            }

            ~ucontext_context_impl()
            {
                if (m_stack) free_stack(m_stack, m_stack_size);
            }

            // Return the size of the reserved stack address space.
            std::ptrdiff_t get_stacksize() const { return m_stack_size; }

            std::ptrdiff_t get_available_stack_space()
            {
# if defined(PIKA_HAVE_THREADS_GET_STACK_POINTER)
                return get_stack_ptr() - reinterpret_cast<std::size_t>(m_stack);
# else
                return (std::numeric_limits<std::ptrdiff_t>::max)();
# endif
            }

            void reset_stack()
            {
                if (m_stack)
                {
                    if (posix::reset_stack(m_stack, static_cast<std::size_t>(m_stack_size)))
                    {
# if defined(PIKA_HAVE_COROUTINE_COUNTERS)
                        increment_stack_unbind_count();
# endif
                    }
                }
            }

            void rebind_stack()
            {
                if (m_stack)
                {
                    // just reset the context stack pointer to its initial value at
                    // the stack start
# if defined(PIKA_HAVE_COROUTINE_COUNTERS)
                    increment_stack_recycle_count();
# endif
                    [[maybe_unused]] int error = PIKA_COROUTINE_MAKE_CONTEXT(
                        &m_ctx, m_stack, m_stack_size, funp_, this, nullptr);
                    PIKA_ASSERT(error == 0);

# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
                    asan_stack_size = m_stack_size;
                    asan_stack_bottom = const_cast<void const*>(m_stack);
# endif
                }
            }

# if defined(PIKA_HAVE_COROUTINE_COUNTERS)
            using counter_type = std::atomic<std::int64_t>;

        private:
            static counter_type& get_stack_unbind_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static counter_type& get_stack_recycle_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static std::uint64_t increment_stack_unbind_count()
            {
                return ++get_stack_unbind_counter();
            }

            static std::uint64_t increment_stack_recycle_count()
            {
                return ++get_stack_recycle_counter();
            }

        public:
            static std::uint64_t get_stack_unbind_count(bool reset)
            {
                return detail::get_and_reset_value(get_stack_unbind_counter(), reset);
            }

            static std::uint64_t get_stack_recycle_count(bool reset)
            {
                return detail::get_and_reset_value(get_stack_recycle_counter(), reset);
            }
# endif

        private:
            // declare m_stack_size first so we can use it to initialize m_stack
            std::ptrdiff_t m_stack_size;
            void* m_stack;
            void (*funp_)(void*);
        };
    }    // namespace detail::posix
}    // namespace pika::threads::coroutines

#else

/**
 * This #else clause is essentially unchanged from the original Google Summer
 * of Code version of Boost.Coroutine, which comments:
 * "Context swapping can be implemented on most posix systems lacking *context
 * using the signaltstack+longjmp trick."
 * This is in fact what the (highly portable) Pth library does, so if you
 * encounter such a system, perhaps the best approach would be to twiddle the
 * #if logic in this header to use the pth.h implementation above.
 */
# error No context implementation for this POSIX system.

#endif
