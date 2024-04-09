//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/thread_pool_helpers.hpp>
#include <pika/topology/cpu_mask.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pika::resource {
    std::size_t get_num_thread_pools() { return get_partitioner().get_num_pools(); }

    std::size_t get_num_threads() { return get_partitioner().get_num_threads(); }

    std::size_t get_num_threads(std::string const& pool_name)
    {
        return get_partitioner().get_num_threads(pool_name);
    }

    std::size_t get_num_threads(std::size_t pool_index)
    {
        return get_partitioner().get_num_threads(pool_index);
    }

    std::size_t get_pool_index(std::string const& pool_name)
    {
        return get_partitioner().get_pool_index(pool_name);
    }

    std::string const& get_pool_name(std::size_t pool_index)
    {
        return get_partitioner().get_pool_name(pool_index);
    }

    pika::threads::detail::thread_pool_base& get_thread_pool(std::string const& pool_name)
    {
        return pika::detail::get_runtime().get_thread_manager().get_pool(pool_name);
    }

    pika::threads::detail::thread_pool_base& get_thread_pool(std::size_t pool_index)
    {
        return get_thread_pool(get_pool_name(pool_index));
    }

    bool pool_exists(std::string const& pool_name)
    {
        return pika::detail::get_runtime().get_thread_manager().pool_exists(pool_name);
    }

    bool pool_exists(std::size_t pool_index)
    {
        return pika::detail::get_runtime().get_thread_manager().pool_exists(pool_index);
    }
}    // namespace pika::resource

namespace pika::threads {
    std::int64_t get_idle_core_count()
    {
        return pika::detail::get_runtime().get_thread_manager().get_idle_core_count();
    }

    detail::mask_type get_idle_core_mask()
    {
        return pika::detail::get_runtime().get_thread_manager().get_idle_core_mask();
    }

    bool enumerate_threads(pika::util::detail::function<bool(detail::thread_id_type)> const& f,
        detail::thread_schedule_state state)
    {
        return pika::detail::get_runtime().get_thread_manager().enumerate_threads(f, state);
    }
}    // namespace pika::threads
