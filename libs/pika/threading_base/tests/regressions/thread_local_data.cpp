//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/async_local.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threading_base.hpp>

#include <atomic>
#include <cstddef>

std::atomic<bool> data_deallocated(false);

struct test_data
{
    test_data() = default;

    ~test_data()
    {
        data_deallocated = true;
    }
};

void test()
{
    pika::threads::thread_id_type id = pika::threads::get_self_id();
    test_data* p = new test_data;
    pika::threads::add_thread_exit_callback(id, [p, id]() {
        pika::threads::thread_id_type id1 = pika::threads::get_self_id();
        PIKA_TEST_EQ(id1, id);

        test_data* p1 =
            reinterpret_cast<test_data*>(pika::threads::get_thread_data(id1));
        PIKA_TEST_EQ(p1, p);

        delete p;
    });
    pika::threads::set_thread_data(id, reinterpret_cast<std::size_t>(p));
}

int pika_main()
{
    pika::async(&test).get();
    PIKA_TEST(data_deallocated);
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
