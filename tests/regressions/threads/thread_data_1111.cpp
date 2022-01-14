//  Copyright (c) 2014 Jeremy Kemp
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test illustrates #1111: pika::threads::get_thread_data always returns zero

#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <memory>

struct thread_data
{
    int thread_num;
};

int get_thread_num()
{
    pika::threads::thread_id_type thread_id = pika::threads::get_self_id();
    thread_data* data = reinterpret_cast<thread_data*>(
        pika::threads::get_thread_data(thread_id));
    PIKA_TEST(data);
    return data ? data->thread_num : 0;
}

int pika_main()
{
    std::unique_ptr<thread_data> data_struct(new thread_data());
    data_struct->thread_num = 42;

    pika::threads::thread_id_type thread_id = pika::threads::get_self_id();
    pika::threads::set_thread_data(
        thread_id, reinterpret_cast<std::size_t>(data_struct.get()));

    PIKA_TEST_EQ(get_thread_num(), 42);

    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    pika::local::init(pika_main, argc, argv);

    return pika::util::report_errors();
}
