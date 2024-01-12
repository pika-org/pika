//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <cstdlib>
#include <string>
#include <vector>

int pika_main()
{
    pika::finalize();
    return EXIT_SUCCESS;
}

int main()
{
    int const argc = 1;
    std::string name = "test";

    {
        char* argv[] = {name.data(), nullptr};

        PIKA_TEST_EQ(pika::init(pika_main, argc, argv), 0);

        pika::start(pika_main, argc, argv);
        PIKA_TEST_EQ(pika::stop(), 0);
    }

    {
        const char* argv[] = {name.data(), nullptr};

        PIKA_TEST_EQ(pika::init(pika_main, argc, argv), 0);

        pika::start(pika_main, argc, argv);
        PIKA_TEST_EQ(pika::stop(), 0);
    }

    {
        char* const argv[] = {name.data(), nullptr};

        PIKA_TEST_EQ(pika::init(pika_main, argc, argv), 0);

        pika::start(pika_main, argc, argv);
        PIKA_TEST_EQ(pika::stop(), 0);
    }

    {
        const char* const argv[] = {name.data(), nullptr};

        PIKA_TEST_EQ(pika::init(pika_main, argc, argv), 0);

        pika::start(pika_main, argc, argv);
        PIKA_TEST_EQ(pika::stop(), 0);
    }

    return 0;
}
