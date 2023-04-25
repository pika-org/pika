//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/function.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#include <pika/threading_base/thread_queue_init_parameters.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace pika::resource {
    class numa_domain;
    class core;
    class pu;

    class partitioner;

    namespace detail {
        class PIKA_EXPORT partitioner;
        void PIKA_EXPORT delete_partitioner();
    }    // namespace detail

    /// May be used anywhere in code and returns a reference to the single,
    /// global resource partitioner.
    PIKA_EXPORT detail::partitioner& get_partitioner();

    /// Returns true if the resource partitioner has been initialized.
    /// Returns false otherwise.
    PIKA_EXPORT bool is_partitioner_valid();

    /// This enumeration describes the modes available when creating a
    /// resource partitioner.
    enum partitioner_mode
    {
        /// Default mode.
        mode_default = 0,
        /// Allow processing units to be oversubscribed, i.e. multiple
        /// worker threads to share a single processing unit.
        mode_allow_oversubscription = 1,
        /// Allow worker threads to be added and removed from thread pools.
        mode_allow_dynamic_pools = 2
    };

    using scheduler_function =
        util::detail::function<std::unique_ptr<pika::threads::detail::thread_pool_base>(
            pika::threads::detail::thread_pool_init_parameters,
            pika::threads::detail::thread_queue_init_parameters)>;

    // Choose same names as in command-line options except with _ instead of
    // -.

    /// This enumeration lists the available scheduling policies (or
    /// schedulers) when creating thread pools.
    enum scheduling_policy
    {
        user_defined = -2,
        unspecified = -1,
        local = 0,
        local_priority_fifo = 1,
        local_priority_lifo = 2,
        static_ = 3,
        static_priority = 4,
        abp_priority_fifo = 5,
        abp_priority_lifo = 6,
        shared_priority = 7,
    };
}    // namespace pika::resource
