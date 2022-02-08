//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once    // prevent multiple inclusions of this header file.

#include <pika/config.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#include <apex_api.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace pika::detail::external_timer {
    inline uint64_t init(const char* thread_name, const uint64_t comm_rank,
        const uint64_t comm_size)
    {
        return apex::init(thread_name, comm_rank, comm_size);
    }

    inline void finalize(void)
    {
        apex::finalize();
    }

    inline void register_thread(const std::string& name)
    {
        apex::register_thread(name);
    }

    inline std::shared_ptr<task_wrapper> new_task(const std::string& name,
        const uint64_t task_id, const std::shared_ptr<task_wrapper> parent_task)
    {
        return apex::new_task(name, task_id, parent_task);
    }

    inline std::shared_ptr<task_wrapper> new_task(uintptr_t address,
        const uint64_t task_id, const std::shared_ptr<task_wrapper> parent_task)
    {
        return apex::new_task(address, task_id, parent_task);
    }

    inline void send(uint64_t tag, uint64_t size, uint64_t target)
    {
        apex::send(tag, size, target);
    }

    inline void recv(uint64_t tag, uint64_t size, uint64_t source_rank,
        uint64_t source_thread)
    {
        apex::recv(tag, size, source_rank, source_thread);
    }

    inline std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper> wrapper, const std::string& name)
    {
        return apex::update_task(wrapper, name);
    }

    inline std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper> wrapper, uintptr_t address)
    {
        return apex::update_task(wrapper, address);
    }

    inline void start(std::shared_ptr<task_wrapper> task_wrapper_ptr)
    {
        apex::start(task_wrapper_ptr);
    }

    inline void stop(std::shared_ptr<task_wrapper> task_wrapper_ptr)
    {
        apex::stop(task_wrapper_ptr);
    }

    inline void yield(std::shared_ptr<task_wrapper> task_wrapper_ptr)
    {
        apex::yield(task_wrapper_ptr);
    }

    PIKA_EXPORT std::shared_ptr<task_wrapper> new_task(
        pika::util::thread_description const& description,
        std::uint32_t parent_locality_id, threads::thread_id_type parent_task);

    PIKA_EXPORT std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper> wrapper,
        pika::util::thread_description const& description);

    // This is a scoped object around task scheduling to measure the time
    // spent executing pika threads
    struct scoped_timer
    {
        explicit scoped_timer(std::shared_ptr<task_wrapper> data_ptr)
          : stopped(false)
          , data_(nullptr)
        {
            // APEX internal actions are not timed. Otherwise, we would end
            // up with recursive timers. So it's possible to have a null
            // task wrapper pointer here.
            if (data_ptr != nullptr)
            {
                data_ = data_ptr;
                pika::detail::external_timer::start(data_);
            }
        }
        ~scoped_timer()
        {
            stop();
        }

        void stop()
        {
            if (!stopped)
            {
                stopped = true;
                // APEX internal actions are not timed. Otherwise, we would
                // end up with recursive timers. So it's possible to have a
                // null task wrapper pointer here.
                if (data_ != nullptr)
                {
                    pika::detail::external_timer::stop(data_);
                }
            }
        }

        void yield()
        {
            if (!stopped)
            {
                stopped = true;
                // APEX internal actions are not timed. Otherwise, we would
                // end up with recursive timers. So it's possible to have a
                // null task wrapper pointer here.
                if (data_ != nullptr)
                {
                    pika::detail::external_timer::yield(data_);
                }
            }
        }

        bool stopped;
        std::shared_ptr<task_wrapper> data_;
    };
}    // namespace pika::detail::external_timer
