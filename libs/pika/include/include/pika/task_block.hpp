//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/parallel/task_block.hpp>

namespace pika {
    using task_cancelled_exception = pika::parallel::task_canceled_exception;
    using pika::parallel::define_task_block;
    using pika::parallel::define_task_block_restore_thread;
    using pika::parallel::task_block;
}    // namespace pika
