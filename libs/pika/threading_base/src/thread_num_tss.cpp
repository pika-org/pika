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

    std::size_t get_worker_thread_num()
    {
        return threads::detail::thread_nums_tss_.global_thread_num;
    }

    std::size_t get_local_worker_thread_num()
    {
        return threads::detail::thread_nums_tss_.local_thread_num;
    }

    std::size_t get_thread_pool_num()
    {
        return threads::detail::thread_nums_tss_.thread_pool_num;
    }

}    // namespace pika
