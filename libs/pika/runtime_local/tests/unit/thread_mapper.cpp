//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/local/runtime.hpp>
#include <pika/modules/testing.hpp>
#include <pika/util/from_string.hpp>

#include <cstddef>
#include <string>
#include <thread>
#include <vector>

void enumerate_threads(std::size_t num_custom_threads)
{
    std::size_t counts[std::size_t(pika::os_thread_type::custom_thread) + 1] = {
        0};

    bool result =
        pika::enumerate_os_threads([&counts](pika::os_thread_data const& data) {
            if (data.type_ != pika::os_thread_type::unknown)
            {
                PIKA_TEST(std::size_t(data.type_) <=
                    std::size_t(pika::os_thread_type::custom_thread));

                ++counts[std::size_t(data.type_)];
                PIKA_TEST(data.label_.find(pika::get_os_thread_type_name(
                             data.type_)) != std::string::npos);
            }
            return true;
        });
    PIKA_TEST(result);

    std::size_t num_workers = pika::get_num_worker_threads();
    PIKA_TEST_EQ(
        counts[std::size_t(pika::os_thread_type::worker_thread)], num_workers);

    PIKA_TEST_EQ(counts[std::size_t(pika::os_thread_type::custom_thread)],
        num_custom_threads);
}

int pika_main()
{
    enumerate_threads(0);

    auto* rt = pika::get_runtime_ptr();

    std::thread t([rt]() {
        pika::register_thread(rt, "custom");
        enumerate_threads(1);
        pika::unregister_thread(rt);
    });
    t.join();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init_params init_args;

    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);

    return pika::util::report_errors();
}
