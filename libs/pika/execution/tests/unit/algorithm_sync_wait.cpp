//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/execution.hpp>
#include <pika/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = pika::execution::experimental;

// NOTE: This is not a conforming sync_wait implementation. It only exists to
// check that the tag_invoke overload is called.
void tag_invoke(ex::sync_wait_t, custom_sender2 s)
{
    s.tag_invoke_overload_called = true;
}

int pika_main()
{
    // Success path
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        ex::sync_wait(custom_sender{
            start_called, connect_called, tag_invoke_overload_called});
        PIKA_TEST(start_called);
        PIKA_TEST(connect_called);
        PIKA_TEST(!tag_invoke_overload_called);
    }

    {
        PIKA_TEST_EQ(ex::sync_wait(ex::just(3)), 3);
    }

    {
        PIKA_TEST_EQ(
            ex::sync_wait(ex::just(custom_type_non_default_constructible{42}))
                .x,
            42);
    }

    {
        PIKA_TEST_EQ(
            ex::sync_wait(
                ex::just(
                    custom_type_non_default_constructible_non_copyable{42}))
                .x,
            42);
    }

    // operator| overload
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        custom_sender{
            start_called, connect_called, tag_invoke_overload_called} |
            ex::sync_wait();
        PIKA_TEST(start_called);
        PIKA_TEST(connect_called);
        PIKA_TEST(!tag_invoke_overload_called);
    }

    {
        PIKA_TEST_EQ(ex::just(3) | ex::sync_wait(), 3);
    }

    // tag_invoke overload
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        ex::sync_wait(custom_sender2{custom_sender{
            start_called, connect_called, tag_invoke_overload_called}});
        PIKA_TEST(!start_called);
        PIKA_TEST(!connect_called);
        PIKA_TEST(tag_invoke_overload_called);
    }

    // Failure path
    {
        bool exception_thrown = false;
        try
        {
            ex::sync_wait(error_sender{});
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }
        PIKA_TEST(exception_thrown);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
