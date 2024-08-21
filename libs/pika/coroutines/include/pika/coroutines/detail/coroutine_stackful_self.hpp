//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/detail/coroutine_impl.hpp>
#include <pika/coroutines/detail/coroutine_self.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/coroutines/thread_id_type.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <cstddef>
#include <exception>
#include <limits>
#include <utility>
#endif

namespace pika::threads::coroutines::detail {

    class coroutine_stackful_self : public coroutine_self
    {
    public:
        explicit coroutine_stackful_self(impl_type* pimpl, coroutine_self* next_self = nullptr)
          : coroutine_self(next_self)
          , pimpl_(pimpl)
        {
        }

        arg_type yield_impl(result_type arg) override
        {
            PIKA_ASSERT(pimpl_);

            this->pimpl_->bind_result(arg);

            {
                reset_self_on_exit on_exit(this);
                this->pimpl_->yield();
            }

            return *pimpl_->args();
        }

        thread_id_type get_thread_id() const override
        {
            PIKA_ASSERT(pimpl_);
            return pimpl_->get_thread_id();
        }

        std::size_t get_thread_phase() const override
        {
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
            PIKA_ASSERT(pimpl_);
            return pimpl_->get_thread_phase();
#else
            return 0;
#endif
        }

        std::ptrdiff_t get_available_stack_space() override
        {
#if defined(PIKA_HAVE_THREADS_GET_STACK_POINTER)
            return pimpl_->get_available_stack_space();
#else
            return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
        }

        std::size_t get_thread_data() const override
        {
            PIKA_ASSERT(pimpl_);
            return pimpl_->get_thread_data();
        }
        std::size_t set_thread_data(std::size_t data) override
        {
            PIKA_ASSERT(pimpl_);
            return pimpl_->set_thread_data(data);
        }

        tss_storage* get_thread_tss_data() override
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            PIKA_ASSERT(pimpl_);
            return pimpl_->get_thread_tss_data(false);
#else
            return nullptr;
#endif
        }

        tss_storage* get_or_create_thread_tss_data() override
        {
#if defined(PIKA_HAVE_THREAD_LOCAL_STORAGE)
            PIKA_ASSERT(pimpl_);
            return pimpl_->get_thread_tss_data(true);
#else
            return nullptr;
#endif
        }

        std::size_t& get_continuation_recursion_count() override
        {
            PIKA_ASSERT(pimpl_);
            return pimpl_->get_continuation_recursion_count();
        }

    private:
        coroutine_impl* get_impl() override { return pimpl_; }
        coroutine_impl* pimpl_;
    };
}    // namespace pika::threads::coroutines::detail
