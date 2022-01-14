//  Copyright (c) 2014-2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/modules/threading_base.hpp>

#include <string>
#include <vector>

namespace pika { namespace util { namespace debug {

    // ------------------------------------------------------------------------
    // return a vector of suspended/other task Ids
    PIKA_EXPORT std::vector<pika::threads::thread_id_type> get_task_ids(
        pika::threads::thread_schedule_state state =
            pika::threads::thread_schedule_state::suspended);

    // ------------------------------------------------------------------------
    // return a vector of thread data structure pointers for suspended tasks
    PIKA_EXPORT std::vector<pika::threads::thread_data*> get_task_data(
        pika::threads::thread_schedule_state state =
            pika::threads::thread_schedule_state::suspended);

    // ------------------------------------------------------------------------
    // return string containing the stack backtrace for suspended tasks
    PIKA_EXPORT std::string suspended_task_backtraces();

}}}    // namespace pika::util::debug
