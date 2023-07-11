//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/thread_num_tss.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace pika {
    namespace threads::detail {
        /// Holds the global and local thread numbers, and the pool number
        /// associated with the thread.
        struct thread_nums
        {
            std::size_t global_thread_num;
            std::size_t local_thread_num;
            std::size_t thread_pool_num;
        };

        static thread_local thread_nums thread_nums_tss_ = {
            std::size_t(-1), std::size_t(-1), std::size_t(-1)};

        std::size_t set_global_thread_num_tss(std::size_t num)
        {
            std::swap(thread_nums_tss_.global_thread_num, num);
            return num;
        }

        std::size_t get_global_thread_num_tss() { return thread_nums_tss_.global_thread_num; }

        std::size_t set_local_thread_num_tss(std::size_t num)
        {
            std::swap(thread_nums_tss_.local_thread_num, num);
            return num;
        }

        std::size_t get_local_thread_num_tss() { return thread_nums_tss_.local_thread_num; }

        std::size_t set_thread_pool_num_tss(std::size_t num)
        {
            std::swap(thread_nums_tss_.thread_pool_num, num);
            return num;
        }

        std::size_t get_thread_pool_num_tss() { return thread_nums_tss_.thread_pool_num; }
    }    // namespace threads::detail

    std::size_t get_worker_thread_num()
    {
        return threads::detail::thread_nums_tss_.global_thread_num;
    }

    std::size_t get_local_worker_thread_num()
    {
        return threads::detail::thread_nums_tss_.local_thread_num;
    }

    std::size_t get_thread_pool_num() { return threads::detail::thread_nums_tss_.thread_pool_num; }

}    // namespace pika
