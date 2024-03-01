////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/config.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/state.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika::detail {
    // return whether thread manager is in the runtime_state described by st
    bool thread_manager_is(runtime_state st)
    {
        pika::runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            // we're probably either starting or stopping
            return st <= runtime_state::starting || st >= runtime_state::stopping;
        }
        return (rt->get_thread_manager().status() == st);
    }
}    // namespace pika::detail
