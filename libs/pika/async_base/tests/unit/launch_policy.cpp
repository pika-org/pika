//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/modules/async_base.hpp>
#include <pika/modules/coroutines.hpp>
#include <pika/modules/testing.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
template <typename Launch>
void test_policy(Launch policy)
{
    PIKA_TEST(policy.priority() == pika::threads::thread_priority::default_);
    PIKA_TEST(policy.stacksize() == pika::threads::thread_stacksize::default_);
    PIKA_TEST(
        policy.hint().mode == pika::threads::thread_schedule_hint_mode::none);
    PIKA_TEST_EQ(policy.hint().hint, std::int16_t(-1));

    policy.set_priority(pika::threads::thread_priority::normal);
    PIKA_TEST(policy.priority() == pika::threads::thread_priority::normal);

    auto p = pika::execution::experimental::with_priority(
        policy, pika::threads::thread_priority::high);
    PIKA_TEST(pika::execution::experimental::get_priority(p) ==
        pika::threads::thread_priority::high);

    policy.set_stacksize(pika::threads::thread_stacksize::medium);
    PIKA_TEST(policy.stacksize() == pika::threads::thread_stacksize::medium);

    auto p1 = pika::execution::experimental::with_stacksize(
        policy, pika::threads::thread_stacksize::small_);
    PIKA_TEST(pika::execution::experimental::get_stacksize(p1) ==
        pika::threads::thread_stacksize::small_);

    pika::threads::thread_schedule_hint hint(0);
    policy.set_hint(hint);
    PIKA_TEST(policy.hint().mode == hint.mode);
    PIKA_TEST_EQ(policy.hint().hint, hint.hint);

    pika::threads::thread_schedule_hint hint1(1);
    auto p2 = pika::execution::experimental::with_hint(policy, hint1);
    PIKA_TEST(pika::execution::experimental::get_hint(p2).mode == hint1.mode);
    PIKA_TEST(pika::execution::experimental::get_hint(p2).hint == hint1.hint);
}

int main()
{
    static_assert(sizeof(pika::launch::async_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(pika::launch::sync_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(pika::launch::deferred_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(pika::launch::fork_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(pika::launch::apply_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(pika::launch) <= sizeof(std::int64_t));

    test_policy(pika::launch::async);
    test_policy(pika::launch::sync);
    test_policy(pika::launch::deferred);
    test_policy(pika::launch::fork);
    test_policy(pika::launch::apply);

    test_policy(pika::launch());
    test_policy(pika::launch(pika::launch::async));
    test_policy(pika::launch(pika::launch::sync));
    test_policy(pika::launch(pika::launch::deferred));
    test_policy(pika::launch(pika::launch::fork));
    test_policy(pika::launch(pika::launch::apply));

    return 0;
}
