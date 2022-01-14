//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/threading_base/detail/get_default_pool.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

// The following implementation has been divided for Linux and Mac OSX
#if (defined(__linux) || defined(__linux__) || defined(linux) ||               \
    defined(__APPLE__))

namespace pika_start {
    // Redefining weak variables defined in pika_main.hpp to facilitate error
    // checking and make sure correct errors are thrown. It is added again
    // to make sure that these variables are defined correctly in cases
    // where pika_main functionalities are not used.
    PIKA_SYMBOL_EXPORT bool is_linked __attribute__((weak)) = false;
    PIKA_SYMBOL_EXPORT bool include_libpika_wrap __attribute__((weak)) =
        false;
}    // namespace pika_start

#endif

namespace pika { namespace threads { namespace detail {
    static get_default_pool_type get_default_pool;

    void set_get_default_pool(get_default_pool_type f)
    {
        get_default_pool = f;
    }

    thread_pool_base* get_self_or_default_pool()
    {
        thread_pool_base* pool = nullptr;
        auto thrd_data = get_self_id_data();
        if (thrd_data)
        {
            pool = thrd_data->get_scheduler_base()->get_parent_pool();
        }
        else if (detail::get_default_pool)
        {
            pool = detail::get_default_pool();
            PIKA_ASSERT(pool);
        }
        else
        {
                    PIKA_THROW_EXCEPTION(invalid_status,
                        "pika::threads::detail::get_self_or_default_pool",
                        "Attempting to register a thread outside the pika "
                        "runtime and no default pool handler is installed. "
                        "Did you mean to run this on an pika thread?");
                }

                return pool;
            }
}}}    // namespace pika::threads::detail
