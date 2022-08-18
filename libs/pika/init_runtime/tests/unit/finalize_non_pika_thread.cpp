//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This test checks that pika::finalize can be called on a non-pika thread.

#include <pika/init.hpp>
#include <pika/testing.hpp>

bool ran_pika_main = false;

int pika_main()
{
    ran_pika_main = true;

    return 0;
}

int main(int argc, char* argv[])
{
    pika::start(pika_main, argc, argv);
    pika::finalize();
    PIKA_TEST_EQ_MSG(pika::stop(), 0, "pika main exited with non-zero status");

    PIKA_TEST(ran_pika_main);

    return pika::util::report_errors();
}
