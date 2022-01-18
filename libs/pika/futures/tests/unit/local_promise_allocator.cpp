//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (C) 2011 Vicente J. Botet Escriba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <memory>

#include "test_allocator.hpp"

int pika_main()
{
    PIKA_TEST_EQ(test_alloc_base::count, 0);
    {
        pika::lcos::local::promise<int> p(
            std::allocator_arg, test_allocator<int>());
        PIKA_TEST_EQ(test_alloc_base::count, 1);
        pika::future<int> f = p.get_future();
        PIKA_TEST_EQ(test_alloc_base::count, 1);
        PIKA_TEST(f.valid());
    }
    PIKA_TEST_EQ(test_alloc_base::count, 0);
    {
        pika::lcos::local::promise<int&> p(
            std::allocator_arg, test_allocator<int>());
        PIKA_TEST_EQ(test_alloc_base::count, 1);
        pika::future<int&> f = p.get_future();
        PIKA_TEST_EQ(test_alloc_base::count, 1);
        PIKA_TEST(f.valid());
    }
    PIKA_TEST_EQ(test_alloc_base::count, 0);
    {
        pika::lcos::local::promise<void> p(
            std::allocator_arg, test_allocator<void>());
        PIKA_TEST_EQ(test_alloc_base::count, 1);
        pika::future<void> f = p.get_future();
        PIKA_TEST_EQ(test_alloc_base::count, 1);
        PIKA_TEST(f.valid());
    }
    PIKA_TEST_EQ(test_alloc_base::count, 0);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init(pika_main, argc, argv);
    return pika::util::report_errors();
}
