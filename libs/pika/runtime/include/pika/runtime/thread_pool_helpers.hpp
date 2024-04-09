//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/runtime/thread_pool_helpers.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#include <pika/topology/cpu_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace pika::resource {
    ///////////////////////////////////////////////////////////////////////////
    /// Return the number of thread pools currently managed by the
    /// \a resource_partitioner
    PIKA_EXPORT std::size_t get_num_thread_pools();

    /// Return the number of threads in all thread pools currently
    /// managed by the \a resource_partitioner
    PIKA_EXPORT std::size_t get_num_threads();

    /// Return the number of threads in the given thread pool currently
    /// managed by the \a resource_partitioner
    PIKA_EXPORT std::size_t get_num_threads(std::string const& pool_name);

    /// Return the number of threads in the given thread pool currently
    /// managed by the \a resource_partitioner
    PIKA_EXPORT std::size_t get_num_threads(std::size_t pool_index);

    /// Return the internal index of the pool given its name.
    PIKA_EXPORT std::size_t get_pool_index(std::string const& pool_name);

    /// Return the name of the pool given its internal index
    PIKA_EXPORT std::string const& get_pool_name(std::size_t pool_index);

    /// Return the name of the pool given its name
    PIKA_EXPORT pika::threads::detail::thread_pool_base& get_thread_pool(
        std::string const& pool_name);

    /// Return the thread pool given its internal index
    PIKA_EXPORT pika::threads::detail::thread_pool_base& get_thread_pool(std::size_t pool_index);

    /// Return true if the pool with the given name exists
    PIKA_EXPORT bool pool_exists(std::string const& pool_name);

    /// Return true if the pool with the given index exists
    PIKA_EXPORT bool pool_exists(std::size_t pool_index);
}    // namespace pika::resource
