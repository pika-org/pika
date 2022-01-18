//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <pika/functional/bind.hpp>
#include <pika/lcos_local/composable_guard.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string>
#include <vector>

typedef std::atomic<int> int_atomic;
int_atomic i1(0), i2(0);
pika::lcos::local::guard_set guards;
std::shared_ptr<pika::lcos::local::guard> l1(new pika::lcos::local::guard());
std::shared_ptr<pika::lcos::local::guard> l2(new pika::lcos::local::guard());

void incr1()
{
    // implicitly lock l1
    int tmp = i1.load();
    PIKA_TEST(i1.compare_exchange_strong(tmp, tmp + 1));
    // implicitly unlock l1
}
void incr2()
{
    // implicitly lock l2
    int tmp = i2.load();
    PIKA_TEST(i2.compare_exchange_strong(tmp, tmp + 1));
    // implicitly unlock l2
}
void both()
{
    // implicitly lock l1 and l2
    int tmp = i1.load();
    PIKA_TEST(i1.compare_exchange_strong(tmp, tmp + 1));
    tmp = i2.load();
    PIKA_TEST(i2.compare_exchange_strong(tmp, tmp + 1));
    // implicitly unlock l1 and l2
}

int increments = 3000;

void check_()
{
    PIKA_TEST(2 * increments == i1 && 2 * increments == i2);
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("increments"))
        increments = vm["increments"].as<int>();

    // create the guard set
    guards.add(l1);
    guards.add(l2);

    for (int i = 0; i < increments; i++)
    {
        // spawn 3 asynchronous tasks
        run_guarded(guards, both);
        run_guarded(*l1, incr1);
        run_guarded(*l2, incr2);
    }

    run_guarded(guards, &::check_);
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::program_options::options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("increments,n",
        pika::program_options::value<int>()->default_value(3000),
        "the number of times to increment the counters");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");
    return pika::util::report_errors();
}
