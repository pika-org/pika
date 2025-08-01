//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/execution.hpp>
#include <pika/testing.hpp>

#include <pika/execution_base/tests/algorithm_test_utils.hpp>

#include <atomic>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;

// Newer versions of stdexec no longer allow tag_invoke customization of algorithms, but we can't
// reliably detect the version of stdexec that dropped support, so we completely exclude this test.
#if !defined(PIKA_HAVE_STDEXEC)
// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename T>
auto tag_invoke(ex::transfer_just_t, scheduler2 s, T&& t)
{
    s.tag_invoke_overload_called.get() = true;

    return ex::transfer_just(
        scheduler{s.schedule_called, s.execute_called, s.tag_invoke_overload_called},
        std::forward<T>(t));
}
#endif

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer_just(scheduler{
            scheduler_schedule_called, scheduler_execute_called, tag_invoke_overload_called});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            3);
        auto f = [](int x) { PIKA_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        int x = 3;
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            x);
        auto f = [](int x) { PIKA_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            custom_type_non_default_constructible{42});
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        custom_type_non_default_constructible x{42};
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            x);
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            custom_type_non_default_constructible_non_copyable{42});
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        custom_type_non_default_constructible_non_copyable x{42};
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            std::move(x));
        auto f = [](auto x) { PIKA_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            std::string("hello"), 3);
        auto f = [](std::string s, int x) {
            PIKA_TEST_EQ(s, std::string("hello"));
            PIKA_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::string str{"hello"};
        int x = 3;
        auto s = ex::transfer_just(scheduler{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            str, x);
        auto f = [](std::string str, int x) {
            PIKA_TEST_EQ(str, std::string("hello"));
            PIKA_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(!tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

#if !defined(PIKA_HAVE_STDEXEC)
    // tag_invoke overload
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer_just(scheduler2{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            3);
        auto f = [](int x) { PIKA_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        int x = 3;
        auto s = ex::transfer_just(scheduler2{scheduler_schedule_called, scheduler_execute_called,
                                       tag_invoke_overload_called},
            x);
        auto f = [](int x) { PIKA_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        PIKA_TEST(set_value_called);
        PIKA_TEST(tag_invoke_overload_called);
        PIKA_TEST(scheduler_schedule_called);
        PIKA_TEST(!scheduler_execute_called);
    }
#endif

    return 0;
}
