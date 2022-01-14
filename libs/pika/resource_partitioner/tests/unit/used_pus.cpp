//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

int pika_main()
{
    std::size_t num_threads = pika::resource::get_num_threads("default");
    pika::threads::thread_pool_base& tp =
        pika::resource::get_thread_pool("default");

    auto used_pu_mask = tp.get_used_processing_units();
    PIKA_TEST_EQ(pika::threads::count(used_pu_mask), num_threads);

    for (std::size_t t = 0; t < num_threads; ++t)
    {
        auto thread_mask = pika::resource::get_partitioner().get_pu_mask(t);
        PIKA_TEST(pika::threads::bit_or(used_pu_mask, thread_mask));
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init_params init_args;
    init_args.cfg = {"pika.os_threads=" +
        std::to_string(((std::min)(std::size_t(4),
            std::size_t(pika::threads::hardware_concurrency()))))};

    // now run the test
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv, init_args), 0);
    return pika::util::report_errors();
}
