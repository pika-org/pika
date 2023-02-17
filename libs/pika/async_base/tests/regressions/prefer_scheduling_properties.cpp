//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/async_base.hpp>
#include <pika/modules/execution.hpp>
#include <pika/modules/properties.hpp>

namespace ex = pika::execution::experimental;

struct scheduler
{
};

int main()
{
    pika::experimental::prefer(
        ex::with_priority, scheduler{}, pika::execution::thread_priority::high);
    pika::experimental::prefer(
        ex::with_stacksize, scheduler{}, pika::execution::thread_stacksize::small_);
    pika::experimental::prefer(ex::with_hint, scheduler{}, pika::execution::thread_schedule_hint{});
    pika::experimental::prefer(ex::with_annotation, scheduler{}, "hello");
    return 0;
}
