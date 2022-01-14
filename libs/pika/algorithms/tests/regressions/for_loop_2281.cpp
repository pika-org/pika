//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <cstddef>
#include <mutex>
#include <set>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    pika::lcos::local::spinlock mtx;
    std::set<pika::thread::id> thread_ids;

    pika::for_loop(pika::execution::par, 0, 100, [&](int) {
        std::lock_guard<pika::lcos::local::spinlock> l(mtx);
        thread_ids.insert(pika::this_thread::get_id());
    });

    PIKA_TEST_LT(std::size_t(1), thread_ids.size());

    thread_ids.clear();

    pika::for_loop_n(pika::execution::par, 0, 100, [&](int) {
        std::lock_guard<pika::lcos::local::spinlock> l(mtx);
        thread_ids.insert(pika::this_thread::get_id());
    });

    PIKA_TEST_LT(std::size_t(1), thread_ids.size());

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"pika.os_threads=4"};

    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
