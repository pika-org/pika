//  Copyright (c)      2022 ETH Zurich
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#include <apex_api.hpp>

#include <cstdint>
#include <memory>

namespace pika::detail::external_timer {
    using apex::finalize;
    using apex::init;
    using apex::new_task;
    using apex::recv;
    using apex::register_thread;
    using apex::send;
    using apex::start;
    using apex::stop;
    using apex::update_task;
    using apex::yield;

    PIKA_EXPORT std::shared_ptr<task_wrapper> new_task(
        pika::util::thread_description const& description,
        std::uint32_t parent_locality_id, threads::thread_id_type parent_task);

    PIKA_EXPORT std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper> wrapper,
        pika::util::thread_description const& description);

    // This is a scoped object around task scheduling to measure the time spent
    // executing pika threads
    struct [[nodiscard]] scoped_timer
    {
        PIKA_EXPORT explicit scoped_timer(
            std::shared_ptr<task_wrapper> data_ptr);
        scoped_timer(scoped_timer&&) = delete;
        scoped_timer(scoped_timer const&) = delete;
        scoped_timer& operator=(scoped_timer&&) = delete;
        scoped_timer& operator=(scoped_timer const&) = delete;
        PIKA_EXPORT ~scoped_timer();

        PIKA_EXPORT void stop();
        PIKA_EXPORT void yield();

    private:
        bool stopped;
        std::shared_ptr<task_wrapper> data_;
    };
}    // namespace pika::detail::external_timer
