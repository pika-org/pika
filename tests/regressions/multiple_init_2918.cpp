//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/bind.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <string>

std::string expected;

int pika_init_test(std::string s, int, char**)
{
    PIKA_TEST_EQ(s, expected);
    return pika::finalize();
}

int main(int argc, char* argv[])
{
    using std::placeholders::_1;
    using std::placeholders::_2;

    expected = "first";
    pika::util::function<int(int, char**)> callback1 =
        pika::util::detail::bind(&pika_init_test, expected, _1, _2);
    pika::init(callback1, argc, argv);

    expected = "second";
    pika::util::function<int(int, char**)> callback2 =
        pika::util::detail::bind(&pika_init_test, expected, _1, _2);
    pika::init(callback2, argc, argv);

    expected = "third";
    pika::util::function<int(int, char**)> callback3 =
        pika::util::detail::bind(&pika_init_test, expected, _1, _2);
    pika::init(callback3, argc, argv);

    return 0;
}
