//  Copyright (c) 2017 Antoine Tran Tan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/executors/execution_policy.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/spmd_block.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <set>
#include <utility>
#include <vector>

std::size_t num_images = 10;
std::size_t iterations = 20;

void bulk_test_function(
    pika::parallel::spmd_block block, std::atomic<std::size_t>* c)
{
    PIKA_TEST_EQ(block.get_num_images(), num_images);
    PIKA_TEST_EQ(block.this_image() < num_images, true);

    // Test sync_all()
    for (std::size_t i = 0, test_count = num_images; i < iterations;
         i++, test_count += num_images)
    {
        ++c[0];
        block.sync_all();
        PIKA_TEST_EQ(c[0], test_count);
        block.sync_all();
    }

    // Test sync_images() with individual values
    std::size_t image_id = block.this_image();

    if ((image_id == 0) || (image_id == 1))
    {
        ++c[1];
    }
    block.sync_images(0, 1);
    if ((image_id == 0) || (image_id == 1))
    {
        PIKA_TEST_EQ(c[1], (std::size_t) 2);
    }

    if ((image_id == 2) || (image_id == 3) || (image_id == 4))
    {
        ++c[2];
    }
    block.sync_images(2, 3, 4);
    if ((image_id == 2) || (image_id == 3) || (image_id == 4))
    {
        PIKA_TEST_EQ(c[2], (std::size_t) 3);
    }

    // Test sync_images() with vector of values
    std::vector<std::size_t> vec_images = {5, 6, 7, 8};

    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        ++c[3];
    }
    block.sync_images(vec_images);
    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        PIKA_TEST_EQ(c[3], (std::size_t) 4);
    }
    block.sync_images(vec_images);
    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        ++c[3];
    }
    block.sync_images(vec_images.begin(), vec_images.end());
    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        PIKA_TEST_EQ(c[3], (std::size_t) 8);
    }
}

int pika_main()
{
    using pika::execution::par;
    using pika::execution::task;

    auto bulk_test = [](pika::parallel::spmd_block block,
                         std::atomic<std::size_t>* c) {
        bulk_test_function(std::move(block), c);
    };

    std::array<std::atomic<std::size_t>, 4> c1, c2, c3;

    for (std::size_t i = 0; i < 4; i++)
    {
        c1[i] = c2[i] = c3[i] = (std::size_t) 0;
    }

    pika::parallel::define_spmd_block(num_images, bulk_test, c1.data());

    std::vector<pika::future<void>> join = pika::parallel::define_spmd_block(
        par(task), num_images, bulk_test, c2.data());

    pika::wait_all(join);

    pika::parallel::define_spmd_block(num_images, bulk_test_function, c3.data());

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv), 0);
    return pika::util::report_errors();
}
