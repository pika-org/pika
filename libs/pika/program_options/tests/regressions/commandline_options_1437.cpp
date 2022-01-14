//  Copyright (c) 2015 Andreas Schaefer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #1437: pika::init() should strip pika-related flags from argv

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

bool invoked_main = false;

int my_pika_main(int argc, char**)
{
    // all pika command line arguments should have been stripped here
    PIKA_TEST_EQ(argc, 1);

    invoked_main = true;
    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    PIKA_TEST_LT(1, argc);

    PIKA_TEST_EQ(pika::local::init(&my_pika_main, argc, argv), 0);
    PIKA_TEST(invoked_main);

    return pika::util::report_errors();
}
