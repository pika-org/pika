//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/synchronization/mutex.hpp>
#include <pika/synchronization/no_mutex.hpp>
#include <pika/synchronization/once.hpp>
#include <pika/synchronization/recursive_mutex.hpp>
#include <pika/thread_support/unlock_guard.hpp>

namespace pika {
    using pika::lcos::local::call_once;
    using pika::lcos::local::mutex;
    using pika::lcos::local::no_mutex;
    using pika::lcos::local::once_flag;
    using pika::lcos::local::recursive_mutex;
    using pika::lcos::local::spinlock;
    using pika::lcos::local::timed_mutex;
    using pika::util::unlock_guard;
}    // namespace pika
