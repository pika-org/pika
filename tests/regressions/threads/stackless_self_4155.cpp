// Copyright (C) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <cstdlib>

void stackless_thread()
{
    PIKA_TEST_NEQ(pika::threads::detail::get_self_id(), pika::threads::detail::invalid_thread_id);
}

int pika_main()
{
    pika::threads::detail::thread_init_data data(
        pika::threads::detail::make_thread_function_nullary(stackless_thread), "stackless_thread",
        pika::execution::thread_priority::default_, pika::execution::thread_schedule_hint(),
        pika::execution::thread_stacksize::nostack);
    pika::threads::detail::register_work(data);
    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    pika::init(pika_main, argc, argv);

    return 0;
}
