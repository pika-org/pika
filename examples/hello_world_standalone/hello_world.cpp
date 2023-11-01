//  Copyright (c) 2012 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>
#include <pika/thread.hpp>

#include <cstdlib>
#include <iostream>

int pika_main()
{
    std::cout << "Hello from pika-thread with id " << pika::this_thread::get_id() << std::endl;

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
