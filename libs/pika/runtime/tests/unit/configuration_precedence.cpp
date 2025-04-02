//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test only starts the runtime. Command line options and environment variables are supplied by
// tests defined in the adjacent CMakeLists.txt, and the output of the program is checked for the
// expected values.

#include <pika/init.hpp>
#include <pika/testing.hpp>

int main(int argc, char** argv)
{
    pika::start(nullptr, argc, argv);
    pika::finalize();
    pika::stop();

    PIKA_TEST(true);

    return 0;
}
