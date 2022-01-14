//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/coroutines/coroutine.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>

#include <atomic>
#include <memory>

namespace pika { namespace threads { namespace detail {

    PIKA_EXPORT thread_state set_thread_state(thread_id_type const& id,
        thread_schedule_state new_state, thread_restart_state new_state_ex,
        thread_priority priority,
        thread_schedule_hint schedulehint = thread_schedule_hint(),
        bool retry_on_active = true, error_code& ec = throws);
}}}    // namespace pika::threads::detail
