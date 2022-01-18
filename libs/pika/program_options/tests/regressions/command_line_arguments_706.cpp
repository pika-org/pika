//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// verify #706 is fixed (`pika::init` removes portions of non-option command
// line arguments before last `=` sign)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

char const* argv[] = {"command_line_argument_test",
    // We need only one thread, this argument should be gone in pika_main
    "--pika:threads=1", "nx=1", "ny=1=5"};

int pika_main(int argc, char** argv_init)
{
    PIKA_TEST_EQ(argc, 3);
    PIKA_TEST_EQ(0, std::strcmp(argv[0], argv_init[0]));
    for (int i = 1; i < argc; ++i)
    {
        PIKA_TEST_EQ(0, std::strcmp(argv[i + 1], argv_init[i]));
    }

    return pika::local::finalize();
}

int main()
{
    PIKA_TEST_EQ(pika::local::init(pika_main, 4, const_cast<char**>(argv)), 0);
    return pika::util::report_errors();
}
