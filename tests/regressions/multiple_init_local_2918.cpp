//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/bind.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <string>

std::string expected;

int pika_init_test(std::string s, int, char**)
{
    PIKA_TEST_EQ(s, expected);
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    using pika::util::placeholders::_1;
    using pika::util::placeholders::_2;

    expected = "first";
    pika::util::function_nonser<int(int, char**)> callback1 =
        pika::util::bind(&pika_init_test, expected, _1, _2);
    pika::local::init(callback1, argc, argv);

    expected = "second";
    pika::util::function_nonser<int(int, char**)> callback2 =
        pika::util::bind(&pika_init_test, expected, _1, _2);
    pika::local::init(callback2, argc, argv);

    expected = "third";
    pika::util::function_nonser<int(int, char**)> callback3 =
        pika::util::bind(&pika_init_test, expected, _1, _2);
    pika::local::init(callback3, argc, argv);

    return 0;
}
