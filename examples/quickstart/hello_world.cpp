//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to execute a pika-thread printing
// "Hello World!" once. That's all.

#include <pika/init.hpp>

#include <cstdlib>
#include <iostream>

int pika_main()
{
    // Say hello to the world!
    std::cout << "Hello World!\n" << std::flush;
    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
