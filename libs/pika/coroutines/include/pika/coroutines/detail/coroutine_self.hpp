//  Copyright (c) 2006, Giovanni P. Deretta
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

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/detail/coroutine_accessor.hpp>
#include <pika/coroutines/detail/coroutine_impl.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/functional/function.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace pika::threads::coroutines::detail {
    class coroutine_self
    {
    public:
        PIKA_NON_COPYABLE(coroutine_self);

    protected:
        // store the current this and write it to the TSS on exit
        struct reset_self_on_exit
        {
            reset_self_on_exit(coroutine_self* self)
              : self_(self)
            {
                set_self(self->next_self_);
            }

            ~reset_self_on_exit() { set_self(self_); }

            coroutine_self* self_;
        };

    public:
        using thread_id_type = pika::threads::detail::thread_id;

        using result_type = std::pair<threads::detail::thread_schedule_state, thread_id_type>;
        using arg_type = pika::threads::detail::thread_restart_state;

        using yield_decorator_type = util::detail::function<arg_type(result_type)>;

        explicit coroutine_self(coroutine_self* next_self)
          : next_self_(next_self)
        {
        }

        arg_type yield(result_type arg = result_type())
        {
            return !yield_decorator_.empty() ? yield_decorator_(std::move(arg)) :
                                               yield_impl(std::move(arg));
        }

        template <typename F>
        yield_decorator_type decorate_yield(F&& f)
        {
            yield_decorator_type tmp(std::forward<F>(f));
            std::swap(tmp, yield_decorator_);
            return tmp;
        }

        yield_decorator_type decorate_yield(yield_decorator_type const& f)
        {
            yield_decorator_type tmp(f);
            std::swap(tmp, yield_decorator_);
            return tmp;
        }

        yield_decorator_type decorate_yield(yield_decorator_type&& f)
        {
            std::swap(f, yield_decorator_);
            return std::move(f);
        }

        yield_decorator_type undecorate_yield()
        {
            yield_decorator_type tmp;
            std::swap(tmp, yield_decorator_);
            return tmp;
        }

        virtual ~coroutine_self() = default;

        virtual arg_type yield_impl(result_type arg) = 0;

        virtual thread_id_type get_thread_id() const = 0;

        virtual std::size_t get_thread_phase() const = 0;

        virtual std::ptrdiff_t get_available_stack_space() = 0;

        virtual std::size_t get_thread_data() const = 0;
        virtual std::size_t set_thread_data(std::size_t data) = 0;

        virtual tss_storage* get_thread_tss_data() = 0;
        virtual tss_storage* get_or_create_thread_tss_data() = 0;

        virtual std::size_t& get_continuation_recursion_count() = 0;

        // access coroutines context object
        using impl_type = coroutine_impl;
        using impl_ptr = impl_type*;

    private:
        friend struct coroutine_accessor;
        virtual impl_ptr get_impl() { return nullptr; }

    public:
        static PIKA_EXPORT coroutine_self*& local_self();

        static void set_self(coroutine_self* self) { local_self() = self; }
        static coroutine_self* get_self() { return local_self(); }

    private:
        yield_decorator_type yield_decorator_;
        coroutine_self* next_self_;
    };

    ////////////////////////////////////////////////////////////////////////////
    struct reset_self_on_exit
    {
        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        reset_self_on_exit(coroutine_self* val, coroutine_self* old_val = nullptr)
          // NOLINTEND(bugprone-easily-swappable-parameters)
          : old_self(old_val)
        {
            coroutine_self::set_self(val);
        }

        ~reset_self_on_exit() { coroutine_self::set_self(old_self); }

        coroutine_self* old_self;
    };

}    // namespace pika::threads::coroutines::detail
