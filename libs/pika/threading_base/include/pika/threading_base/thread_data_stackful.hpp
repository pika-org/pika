//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/execution_agent.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_init_data.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

#include <pika/config/warnings_prefix.hpp>

namespace pika::threads::detail {
    ///////////////////////////////////////////////////////////////////////////
    /// A \a thread is the representation of a ParalleX thread. It's a first
    /// class object in ParalleX. In our implementation this is a user level
    /// thread running on top of one of the OS threads spawned by the \a
    /// thread-manager.
    ///
    /// A \a thread encapsulates:
    ///  - A thread status word (see the functions \a thread#get_state and
    ///    \a thread#set_state)
    ///  - A function to execute (the thread function)
    ///  - A frame (in this implementation this is a block of memory used as
    ///    the threads stack)
    ///  - A block of registers (not implemented yet)
    ///
    /// Generally, \a threads are not created or executed directly. All
    /// functionality related to the management of \a threads is
    /// implemented by the thread-manager.
    class PIKA_EXPORT thread_data_stackful : public thread_data
    {
    private:
        // Avoid warning about using 'this' in initializer list
        thread_data* this_() { return this; }

        static pika::detail::internal_allocator<thread_data_stackful> thread_alloc_;

    public:
        PIKA_FORCEINLINE coroutine_type::result_type call(
            pika::execution::this_thread::detail::agent_storage* agent_storage)
        {
            PIKA_ASSERT(get_state().state() == thread_schedule_state::active);
            PIKA_ASSERT(this == coroutine_.get_thread_id().get());

            pika::execution::this_thread::detail::reset_agent ctx(agent_storage, agent_);
            return coroutine_(set_state_ex(thread_restart_state::signaled));
        }

#if defined(PIKA_DEBUG)
        thread_id_type get_thread_id() const override
        {
            PIKA_ASSERT(this == coroutine_.get_thread_id().get());
            return this->thread_data::get_thread_id();
        }
#endif
#if defined(PIKA_HAVE_THREAD_PHASE_INFORMATION)
        std::size_t get_thread_phase() const noexcept override
        {
            return coroutine_.get_thread_phase();
        }
#endif

        std::size_t get_thread_data() const override { return coroutine_.get_thread_data(); }

        std::size_t set_thread_data(std::size_t data) override
        {
            return coroutine_.set_thread_data(data);
        }

        void init() override { coroutine_.init(); }

        void rebind(thread_init_data& init_data) override
        {
            this->thread_data::rebind_base(init_data);

            coroutine_.rebind(std::move(init_data.func), thread_id_type(this));

            PIKA_ASSERT(coroutine_.is_ready());
        }

        thread_data_stackful(thread_init_data& init_data, void* queue, std::ptrdiff_t stacksize,
            thread_id_addref addref)
          : thread_data(init_data, queue, stacksize, false, addref)
          , coroutine_(std::move(init_data.func), thread_id_type(this_()), stacksize)
          , agent_(coroutine_.impl())
        {
            PIKA_ASSERT(coroutine_.is_ready());
        }

        ~thread_data_stackful();

        static inline thread_data* create(thread_init_data& init_data, void* queue,
            std::ptrdiff_t stacksize, thread_id_addref addref = thread_id_addref::yes);

        void destroy() override
        {
            this->~thread_data_stackful();
            thread_alloc_.deallocate(this, 1);
        }

    private:
        coroutine_type coroutine_;
        execution_agent agent_;
    };

    ////////////////////////////////////////////////////////////////////////////
    inline thread_data* thread_data_stackful::create(
        thread_init_data& data, void* queue, std::ptrdiff_t stacksize, thread_id_addref addref)
    {
        thread_data* p = thread_alloc_.allocate(1);
        new (p) thread_data_stackful(data, queue, stacksize, addref);
        return p;
    }
}    // namespace pika::threads::detail

#include <pika/config/warnings_suffix.hpp>
