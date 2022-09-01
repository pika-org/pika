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
namespace pika { namespace threads {
    // return whether thread manager is in the state described by st
    bool thread_manager_is(state st)
    {
        pika::runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            // we're probably either starting or stopping
            return st <= state_starting || st >= state_stopping;
        }
        return (rt->get_thread_manager().status() == st);
    }
    bool thread_manager_is_at_least(state st)
    {
        pika::runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            // we're probably either starting or stopping
            return false;
        }
        return (rt->get_thread_manager().status() >= st);
    }
}}    // namespace pika::threads
