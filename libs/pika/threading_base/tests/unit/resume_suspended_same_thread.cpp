//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that threads that are suspended get rescheduled on the
// same thread where they were suspended

#include <pika/assert.hpp>
#include <pika/condition_variable.hpp>
#include <pika/init.hpp>
#include <pika/mutex.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <utility>

struct test_data
{
    pika::mutex mtx{};
    pika::condition_variable cond{};
    bool notified{false};
};

int pika_main()
{
    PIKA_ASSERT(pika::get_num_worker_threads() >= 2);

    for (std::size_t i = 0; i < 100; ++i)
    {
        std::cerr << i << "\n";
        std::shared_ptr<test_data> d = std::make_shared<test_data>();

        pika::jthread t1([d] {
            std::unique_lock l{d->mtx};
            auto t = pika::get_worker_thread_num();
            d->cond.wait(l, [&] { return d->notified; });
            PIKA_TEST_EQ(t, pika::get_worker_thread_num());
        });
        pika::jthread t2([d = std::move(d)] {
            std::unique_lock l{d->mtx};
            d->notified = true;
            d->cond.notify_one();
        });
    }

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    pika::init_params init_args;
    // We use the static scheduler to ensure that a thread scheduled on a
    // particular worker thread actually run on that worker thread
    init_args.cfg = {"pika.scheduler=static"};

    PIKA_TEST_EQ(pika::init(pika_main, argc, argv, init_args), 0);

    return 0;
}
