//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This test checks that the runtime takes into account suspended threads before
// initiating full shutdown.

#include <pika/local/config.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/testing.hpp>

#include <chrono>

int pika_main()
{
    pika::apply(
        [] { pika::this_thread::sleep_for(std::chrono::milliseconds(500)); });

    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv), 0);

    return pika::util::report_errors();
}
