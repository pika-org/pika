//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1702: Shared_mutex does
// not compile with no_mutex cond_var

#include <pika/init.hpp>
#include <pika/synchronization/shared_mutex.hpp>
#include <pika/testing.hpp>

#include <mutex>
#include <shared_mutex>

int pika_main()
{
    using shared_mutex_type = pika::shared_mutex;

    int data = 0;
    shared_mutex_type mtx;

    {
        std::unique_lock<shared_mutex_type> l(mtx);
        data = 42;
    }

    {
        std::shared_lock<shared_mutex_type> l(mtx);
        int i = data;
        PIKA_UNUSED(i);
    }

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return 0;
}
