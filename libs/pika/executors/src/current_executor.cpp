//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/executors/current_executor.hpp>

namespace pika { namespace threads {
    parallel::execution::current_executor get_executor(
        thread_id_type const& id, error_code& ec)
    {
        if (PIKA_UNLIKELY(!id))
        {
            PIKA_THROWS_IF(ec, null_thread_id, "pika::threads::get_executor",
                "null thread id encountered");
            return parallel::execution::current_executor();
        }

        if (&ec != &throws)
            ec = make_success_code();

        return parallel::execution::current_executor(
            get_thread_id_data(id)->get_scheduler_base()->get_parent_pool());
    }
}}    // namespace pika::threads

namespace pika { namespace this_thread {
    parallel::execution::current_executor get_executor(error_code& ec)
    {
        return threads::get_executor(threads::get_self_id(), ec);
    }
}}    // namespace pika::this_thread
