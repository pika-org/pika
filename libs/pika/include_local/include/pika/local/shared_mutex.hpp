//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/synchronization/lock_types.hpp>
#include <pika/synchronization/shared_mutex.hpp>

namespace pika {
    using pika::lcos::local::shared_mutex;
    using pika::lcos::local::upgrade_lock;
    using pika::lcos::local::upgrade_to_unique_lock;
}    // namespace pika
