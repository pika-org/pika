//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <pika/serialization/serialize.hpp>

struct A
{
};

int main()
{
    std::vector<char> vector;
    {
        pika::serialization::output_archive oar(vector);
        A a;
        oar << a;
    }
    {
        pika::serialization::input_archive iar(vector);
        A a;
        iar >> a;
    }

    return 0;
}
