//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <cstdlib>

int pika_main()
{
    PIKA_TEST(pika::is_runtime_initialized());
    pika::finalize();
    PIKA_TEST(pika::is_runtime_initialized());

    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    {
        PIKA_TEST(!pika::is_runtime_initialized());
        pika::start(argc, argv);
        PIKA_TEST(pika::is_runtime_initialized());
        pika::finalize();
        PIKA_TEST(pika::is_runtime_initialized());
        pika::stop();
        PIKA_TEST(!pika::is_runtime_initialized());
    }

    {
        PIKA_TEST(!pika::is_runtime_initialized());
        pika::start(pika_main, argc, argv);
        PIKA_TEST(pika::is_runtime_initialized());
        pika::stop();
        PIKA_TEST(!pika::is_runtime_initialized());
    }

    {
        PIKA_TEST(!pika::is_runtime_initialized());
        pika::init(pika_main, argc, argv);
        PIKA_TEST(!pika::is_runtime_initialized());
    }

    return EXIT_SUCCESS;
}
