//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/assert.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

std::size_t const max_threads = (std::min)(
    std::size_t(4), std::size_t(pika::threads::hardware_concurrency()));

int pika_main()
{
    PIKA_TEST_EQ(std::size_t(max_threads), pika::resource::get_num_threads());
    PIKA_TEST_EQ(std::size_t(max_threads), pika::resource::get_num_threads(0));
    PIKA_TEST_EQ(std::size_t(1), pika::resource::get_num_thread_pools());
    PIKA_TEST_EQ(std::size_t(0), pika::resource::get_pool_index("default"));
    PIKA_TEST_EQ(std::string("default"), pika::resource::get_pool_name(0));

    {
        pika::threads::thread_pool_base& pool =
            pika::resource::get_thread_pool(0);
        PIKA_TEST_EQ(std::size_t(0), pool.get_pool_index());
        PIKA_TEST_EQ(std::string("default"), pool.get_pool_name());
        PIKA_TEST_EQ(std::size_t(0), pool.get_thread_offset());
    }

    {
        pika::threads::thread_pool_base& pool =
            pika::resource::get_thread_pool("default");
        PIKA_TEST_EQ(std::size_t(0), pool.get_pool_index());
        PIKA_TEST_EQ(std::string("default"), pool.get_pool_name());
        PIKA_TEST_EQ(std::size_t(0), pool.get_thread_offset());
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_ASSERT(max_threads >= 2);

    pika::local::init_params init_args;
    init_args.cfg = {"pika.os_threads=" +
        std::to_string(((std::min)(std::size_t(4),
            std::size_t(pika::threads::hardware_concurrency()))))};

    // now run the test
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
    return pika::util::report_errors();
}
