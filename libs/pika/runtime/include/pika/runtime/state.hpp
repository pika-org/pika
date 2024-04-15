////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/config.hpp>
#include <pika/runtime/state.hpp>
#include <pika/threading_base/scheduler_state.hpp>

namespace pika::detail {
    // return whether thread manager is in the state described by 'mask'
    PIKA_EXPORT bool thread_manager_is(runtime_state st);
}    // namespace pika::detail
