////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/local/config.hpp>
#include <pika/modules/threadmanager.hpp>
#include <pika/runtime_local/runtime_local.hpp>
#include <pika/runtime_local/state.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace threads {
    // return whether thread manager is in the state described by st
    bool threadmanager_is(state st)
    {
        pika::runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            // we're probably either starting or stopping
            return st <= state_starting || st >= state_stopping;
        }
        return (rt->get_thread_manager().status() == st);
    }
    bool threadmanager_is_at_least(state st)
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
