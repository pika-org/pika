//  Copyright (c) 2018 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/serialization.hpp>
#include <pika/modules/testing.hpp>

#include <vector>

struct A
{
    double a;
    int p;

    void serialize(int, unsigned)
    {
        // 3rd-party logic...
    }
};

template <class Ar>
void serialize(Ar& ar, A& a, unsigned)
{
    ar& a.a;
    ar& a.p;
}

int main()
{
    std::vector<char> vector;
    {
        pika::serialization::output_archive oar(vector);
        A a{2., 4};
        oar << a;
    }

    {
        A a;
        pika::serialization::input_archive iar(vector);
        iar >> a;
        PIKA_TEST_EQ(a.a, 2.);
        PIKA_TEST_EQ(a.p, 4);
    }

    return 0;
}
