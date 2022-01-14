//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2017 Agustin Berge
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/errors/exception.hpp>
#include <pika/errors/throw_exception.hpp>

#include <pika/modules/testing.hpp>

#include <exception>
#include <thread>

void throw_always()
{
    PIKA_THROW_EXCEPTION(pika::no_success, "throw_always", "simulated error");
}

std::exception_ptr test_transport()
{
    std::exception_ptr ptr;
    try
    {
        throw_always();
    }
    catch (...)
    {
        ptr = std::current_exception();
        PIKA_TEST_EQ(
            pika::get_error_what(ptr), "simulated error: pika(no_success)");
        PIKA_TEST_EQ(pika::get_error_function_name(ptr), "throw_always");
    }

    return ptr;
}

int main()
{
    bool exception_caught = false;

    try
    {
        throw_always();
    }
    catch (...)
    {
        exception_caught = true;
        auto ptr = std::current_exception();
        PIKA_TEST_EQ(
            pika::get_error_what(ptr), "simulated error: pika(no_success)");
        PIKA_TEST_EQ(pika::get_error_function_name(ptr), "throw_always");
    }
    PIKA_TEST(exception_caught);

    exception_caught = false;
    try
    {
        throw_always();
    }
    catch (pika::exception& e)
    {
        exception_caught = true;
        PIKA_TEST_EQ(pika::get_error_what(e), "simulated error: pika(no_success)");
        PIKA_TEST_EQ(pika::get_error_function_name(e), "throw_always");
    }
    PIKA_TEST(exception_caught);

    exception_caught = false;
    try
    {
        throw_always();
    }
    catch (pika::exception_info& e)
    {
        exception_caught = true;
        PIKA_TEST_EQ(pika::get_error_what(e), "simulated error: pika(no_success)");
        PIKA_TEST_EQ(pika::get_error_function_name(e), "throw_always");
    }
    PIKA_TEST(exception_caught);

    {
        std::exception_ptr ptr = test_transport();
        PIKA_TEST(ptr);
        PIKA_TEST_EQ(
            pika::get_error_what(ptr), "simulated error: pika(no_success)");
        PIKA_TEST_EQ(pika::get_error_function_name(ptr), "throw_always");
    }

    {
        std::exception_ptr ptr;
        std::thread t([&ptr]() { ptr = test_transport(); });
        t.join();
        PIKA_TEST(ptr);
        PIKA_TEST_EQ(
            pika::get_error_what(ptr), "simulated error: pika(no_success)");
        PIKA_TEST_EQ(pika::get_error_function_name(ptr), "throw_always");
    }

    return pika::util::report_errors();
}
