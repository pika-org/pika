//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>

#include <pika/assert.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/coroutines/detail/coroutine_impl.hpp>
#include <pika/coroutines/detail/coroutine_stackful_self.hpp>

#include <cstddef>
#include <exception>
#include <utility>

namespace pika::threads::coroutines::detail {
#if defined(PIKA_DEBUG)
    coroutine_impl::~coroutine_impl()
    {
        PIKA_ASSERT(!m_fun);    // functor should have been reset by now
    }
#endif

    void coroutine_impl::operator()() noexcept
    {
        using context_exit_status = super_type::context_exit_status;
        context_exit_status status = super_type::ctx_not_exited;

        // yield value once the thread function has finished executing
        result_type result_last(
            threads::detail::thread_schedule_state::unknown, threads::detail::invalid_thread_id);

        // loop as long this coroutine has been rebound
        do {
#if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            finish_switch_fiber(nullptr, m_caller);
#endif
            std::exception_ptr tinfo;
            {
                coroutine_self* old_self = coroutine_self::get_self();
                coroutine_stackful_self self(this, old_self);
                reset_self_on_exit on_exit(&self, old_self);
                try
                {
                    result_last = m_fun(*this->args());
                    PIKA_ASSERT(
                        result_last.first == threads::detail::thread_schedule_state::terminated);
                    status = super_type::ctx_exited_return;
                }
                // NOLINTNEXTLINE(bugprone-empty-catch)
                catch (...)
                {
                    status = super_type::ctx_exited_abnormally;
                    tinfo = std::current_exception();
                }

                // Reset early as the destructors may still yield.
                this->reset_tss();
                this->reset();

                // return value to other side of the fence
                this->bind_result(result_last);
            }

            this->do_return(status, std::move(tinfo));
        } while (this->m_state == super_type::ctx_running);

        // should not get here, never
        PIKA_ASSERT(this->m_state == super_type::ctx_running);
    }
}    // namespace pika::threads::coroutines::detail
