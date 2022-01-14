//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example only creates a resource partitioner without using it. It is
// intended for inclusion in the documentation.

//[body
#include <pika/local/init.hpp>

int pika_main()
{
    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    // Setup the init parameters
    pika::local::init_params init_args;
    pika::local::init(pika_main, argc, argv, init_args);
}
//body]
