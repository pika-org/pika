//  Copyright (c) 2017-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <array>
#include <chrono>
#include <stdexcept>
#include <thread>

int make_int_slowly()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

pika::future<int> make_future()
{
    pika::lcos::local::packaged_task<int()> task(make_int_slowly);
    return task.get_future();
}

void test_wait_any()
{
    {
        std::array<pika::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] = make_future();

        pika::wait_any_nothrow(future_array);

        int count = 0;
        for (auto& f : future_array)
        {
            if (f.is_ready())
            {
                ++count;
            }
        }
        PIKA_TEST_NEQ(count, 0);
    }
    {
        std::array<pika::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            pika::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            pika::wait_any_nothrow(future_array);

            int count = 0;
            for (auto& f : future_array)
            {
                if (f.is_ready())
                {
                    ++count;
                }
            }
            PIKA_TEST_NEQ(count, 0);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!caught_exception);
    }
    {
        std::array<pika::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            pika::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            pika::wait_any(future_array);
            PIKA_TEST(false);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(caught_exception);
    }
}

void test_wait_any_n()
{
    {
        std::array<pika::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] = make_future();

        pika::wait_any_n_nothrow(future_array.begin(), 2);

        int count = 0;
        for (auto& f : future_array)
        {
            if (f.is_ready())
            {
                ++count;
            }
        }
        PIKA_TEST_NEQ(count, 0);
    }
    {
        std::array<pika::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            pika::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            pika::wait_any_n_nothrow(future_array.begin(), 2);

            int count = 0;
            for (auto& f : future_array)
            {
                if (f.is_ready())
                {
                    ++count;
                }
            }
            PIKA_TEST_NEQ(count, 0);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(!caught_exception);
    }
    {
        std::array<pika::future<int>, 2> future_array;
        future_array[0] = make_future();
        future_array[1] =
            pika::make_exceptional_future<int>(std::runtime_error(""));

        bool caught_exception = false;
        try
        {
            pika::wait_any_n(future_array.begin(), 2);
            PIKA_TEST(false);
        }
        catch (std::runtime_error const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(caught_exception);
    }
}

int pika_main()
{
    test_wait_any();
    test_wait_any_n();
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv), 0);
    return pika::util::report_errors();
}
