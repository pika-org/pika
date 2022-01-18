// Copyright (C) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

void stackless_thread()
{
    PIKA_TEST_NEQ(pika::threads::get_self_id(), pika::threads::invalid_thread_id);
}

int pika_main()
{
    pika::threads::thread_init_data data(
        pika::threads::make_thread_function_nullary(stackless_thread),
        "stackless_thread", pika::threads::thread_priority::default_,
        pika::threads::thread_schedule_hint(),
        pika::threads::thread_stacksize::nostack);
    pika::threads::register_work(data);
    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    pika::local::init(pika_main, argc, argv);

    return pika::util::report_errors();
}
