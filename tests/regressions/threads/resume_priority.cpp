//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/pika.hpp>
#include <pika/pika_init.hpp>
#include <pika/modules/testing.hpp>

void low_priority()
{
    PIKA_TEST_EQ(
        pika::threads::thread_priority::low, pika::this_thread::get_priority());
    pika::this_thread::yield();
    PIKA_TEST_EQ(
        pika::threads::thread_priority::low, pika::this_thread::get_priority());
}
PIKA_DECLARE_ACTION(low_priority)
PIKA_ACTION_HAS_LOW_PRIORITY(low_priority_action)
PIKA_PLAIN_ACTION(low_priority)

void normal_priority()
{
    PIKA_TEST_EQ(pika::threads::thread_priority::normal,
        pika::this_thread::get_priority());
    pika::this_thread::yield();
    PIKA_TEST_EQ(pika::threads::thread_priority::normal,
        pika::this_thread::get_priority());
}
PIKA_DECLARE_ACTION(normal_priority)
PIKA_ACTION_HAS_NORMAL_PRIORITY(normal_priority_action)
PIKA_PLAIN_ACTION(normal_priority)

void high_priority()
{
    PIKA_TEST_EQ(
        pika::threads::thread_priority::high, pika::this_thread::get_priority());
    pika::this_thread::yield();
    PIKA_TEST_EQ(
        pika::threads::thread_priority::high, pika::this_thread::get_priority());
}
PIKA_DECLARE_ACTION(high_priority)
PIKA_ACTION_HAS_HIGH_PRIORITY(high_priority_action)
PIKA_PLAIN_ACTION(high_priority)

void high_recursive_priority()
{
    PIKA_TEST_EQ(pika::threads::thread_priority::high_recursive,
        pika::this_thread::get_priority());
    pika::this_thread::yield();
    PIKA_TEST_EQ(pika::threads::thread_priority::high_recursive,
        pika::this_thread::get_priority());
}
PIKA_DECLARE_ACTION(high_recursive_priority)
PIKA_ACTION_HAS_HIGH_RECURSIVE_PRIORITY(high_recursive_priority_action)
PIKA_PLAIN_ACTION(high_recursive_priority)

int pika_main()
{
    low_priority_action()(pika::find_here());
    normal_priority_action()(pika::find_here());
    high_priority_action()(pika::find_here());
    high_recursive_priority_action()(pika::find_here());
    return pika::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ(0, pika::init(argc, argv));
    return pika::util::report_errors();
}
#endif
