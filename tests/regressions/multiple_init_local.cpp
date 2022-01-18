//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #565

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

int invoked_init = 0;

int pika_main()
{
    ++invoked_init;
    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    // Everything is fine on the first call
    pika::local::init(pika_main, argc, argv);
    PIKA_TEST_EQ(invoked_init, 1);

    // Segfault on the call, now fixed
    pika::local::init(pika_main, argc, argv);
    PIKA_TEST_EQ(invoked_init, 2);

    return pika::util::report_errors();
}
