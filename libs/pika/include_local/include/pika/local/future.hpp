//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/modules/async_base.hpp>
#include <pika/modules/async_combinators.hpp>
#include <pika/modules/async_local.hpp>
#include <pika/modules/execution.hpp>
#include <pika/modules/futures.hpp>
#include <pika/modules/lcos_local.hpp>

namespace pika {
    using pika::lcos::local::packaged_task;
}
