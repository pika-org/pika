//  Taken from the Boost.Bind library
//  protect_test.cpp
//
//  Copyright (c) 2009 Steven Watanabe
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <pika/functional/bind.hpp>
#include <pika/functional/protect.hpp>

namespace placeholders = pika::util::placeholders;

#include <pika/modules/testing.hpp>

int f(int x)
{
    return x;
}

int& g(int& x)
{
    return x;
}

template <class T>
const T& constify(const T& arg)
{
    return arg;
}

int main()
{
    int i[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    // non-const

    // test nullary
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, 1))(), 1);

    // test lvalues

    PIKA_TEST_EQ(
        &pika::util::protect(pika::util::bind(g, placeholders::_1))(i[0]), &i[0]);

    PIKA_TEST_EQ(
        &pika::util::protect(pika::util::bind(g, placeholders::_1))(i[0], i[1]),
        &i[0]);
    PIKA_TEST_EQ(
        &pika::util::protect(pika::util::bind(g, placeholders::_2))(i[0], i[1]),
        &i[1]);

    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_1))(
                    i[0], i[1], i[2]),
        &i[0]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_2))(
                    i[0], i[1], i[2]),
        &i[1]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_3))(
                    i[0], i[1], i[2]),
        &i[2]);

    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_1))(
                    i[0], i[1], i[2], i[3]),
        &i[0]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_2))(
                    i[0], i[1], i[2], i[3]),
        &i[1]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_3))(
                    i[0], i[1], i[2], i[3]),
        &i[2]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_4))(
                    i[0], i[1], i[2], i[3]),
        &i[3]);

    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_1))(
                    i[0], i[1], i[2], i[3], i[4]),
        &i[0]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_2))(
                    i[0], i[1], i[2], i[3], i[4]),
        &i[1]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_3))(
                    i[0], i[1], i[2], i[3], i[4]),
        &i[2]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_4))(
                    i[0], i[1], i[2], i[3], i[4]),
        &i[3]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_5))(
                    i[0], i[1], i[2], i[3], i[4]),
        &i[4]);

    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_1))(
                    i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[0]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_2))(
                    i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[1]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_3))(
                    i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[2]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_4))(
                    i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[3]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_5))(
                    i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[4]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_6))(
                    i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[5]);

    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_1))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[0]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_2))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[1]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_3))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[2]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_4))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[3]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_5))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[4]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_6))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[5]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_7))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[6]);

    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_1))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[0]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_2))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[1]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_3))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[2]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_4))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[3]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_5))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[4]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_6))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[5]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_7))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[6]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_8))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[7]);

    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_1))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[0]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_2))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[1]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_3))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[2]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_4))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[3]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_5))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[4]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_6))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[5]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_7))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[6]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_8))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[7]);
    PIKA_TEST_EQ(&pika::util::protect(pika::util::bind(g, placeholders::_9))(
                    i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[8]);

    // test rvalues

    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_1))(0), 0);

    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_1))(0, 1), 0);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_2))(0, 1), 1);

    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_1))(0, 1, 2), 0);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_2))(0, 1, 2), 1);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_3))(0, 1, 2), 2);

    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_1))(0, 1, 2, 3),
        0);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_2))(0, 1, 2, 3),
        1);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_3))(0, 1, 2, 3),
        2);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_4))(0, 1, 2, 3),
        3);
    //
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_1))(0, 1, 2, 3, 4),
        0);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_2))(0, 1, 2, 3, 4),
        1);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_3))(0, 1, 2, 3, 4),
        2);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_4))(0, 1, 2, 3, 4),
        3);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_5))(0, 1, 2, 3, 4),
        4);

    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_1))(
                    0, 1, 2, 3, 4, 5),
        0);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_2))(
                    0, 1, 2, 3, 4, 5),
        1);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_3))(
                    0, 1, 2, 3, 4, 5),
        2);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_4))(
                    0, 1, 2, 3, 4, 5),
        3);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_5))(
                    0, 1, 2, 3, 4, 5),
        4);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_6))(
                    0, 1, 2, 3, 4, 5),
        5);

    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_1))(
                    0, 1, 2, 3, 4, 5, 6),
        0);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_2))(
                    0, 1, 2, 3, 4, 5, 6),
        1);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_3))(
                    0, 1, 2, 3, 4, 5, 6),
        2);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_4))(
                    0, 1, 2, 3, 4, 5, 6),
        3);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_5))(
                    0, 1, 2, 3, 4, 5, 6),
        4);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_6))(
                    0, 1, 2, 3, 4, 5, 6),
        5);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_7))(
                    0, 1, 2, 3, 4, 5, 6),
        6);

    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_1))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        0);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_2))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        1);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_3))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        2);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_4))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        3);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_5))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        4);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_6))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        5);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_7))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        6);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_8))(
                    0, 1, 2, 3, 4, 5, 6, 7),
        7);

    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_1))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        0);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_2))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        1);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_3))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        2);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_4))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        3);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_5))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        4);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_6))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        5);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_7))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        6);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_8))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        7);
    PIKA_TEST_EQ(pika::util::protect(pika::util::bind(f, placeholders::_9))(
                    0, 1, 2, 3, 4, 5, 6, 7, 8),
        8);

    // test mixed perfect forwarding
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_1))(i[0], 1), 0);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_2))(i[0], 1), 1);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_1))(0, i[1]), 0);
    PIKA_TEST_EQ(
        pika::util::protect(pika::util::bind(f, placeholders::_2))(0, i[1]), 1);

    // const

    // test nullary
    PIKA_TEST_EQ(
        constify(constify(pika::util::protect(pika::util::bind(f, 1))))(), 1);

    // test lvalues
    PIKA_TEST_EQ(&constify(constify(pika::util::protect(
                    pika::util::bind(g, placeholders::_1))))(i[0]),
        &i[0]);

    PIKA_TEST_EQ(&constify(constify(pika::util::protect(
                    pika::util::bind(g, placeholders::_1))))(i[0], i[1]),
        &i[0]);
    PIKA_TEST_EQ(&constify(constify(pika::util::protect(
                    pika::util::bind(g, placeholders::_2))))(i[0], i[1]),
        &i[1]);

    PIKA_TEST_EQ(&constify(constify(pika::util::protect(
                    pika::util::bind(g, placeholders::_1))))(i[0], i[1], i[2]),
        &i[0]);
    PIKA_TEST_EQ(&constify(constify(pika::util::protect(
                    pika::util::bind(g, placeholders::_2))))(i[0], i[1], i[2]),
        &i[1]);
    PIKA_TEST_EQ(&constify(constify(pika::util::protect(
                    pika::util::bind(g, placeholders::_3))))(i[0], i[1], i[2]),
        &i[2]);

    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_1)))(i[0], i[1], i[2], i[3]),
        &i[0]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_2)))(i[0], i[1], i[2], i[3]),
        &i[1]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_3)))(i[0], i[1], i[2], i[3]),
        &i[2]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_4)))(i[0], i[1], i[2], i[3]),
        &i[3]);

    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_1)))(i[0], i[1], i[2], i[3], i[4]),
        &i[0]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_2)))(i[0], i[1], i[2], i[3], i[4]),
        &i[1]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_3)))(i[0], i[1], i[2], i[3], i[4]),
        &i[2]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_4)))(i[0], i[1], i[2], i[3], i[4]),
        &i[3]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_5)))(i[0], i[1], i[2], i[3], i[4]),
        &i[4]);

    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_1)))(i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[0]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_2)))(i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[1]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_3)))(i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[2]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_4)))(i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[3]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_5)))(i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[4]);
    PIKA_TEST_EQ(&constify(pika::util::protect(pika::util::bind(
                    g, placeholders::_6)))(i[0], i[1], i[2], i[3], i[4], i[5]),
        &i[5]);

    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_1)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[0]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_2)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[1]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_3)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[2]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_4)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[3]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_5)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[4]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_6)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[5]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_7)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6]),
        &i[6]);

    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_1)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[0]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_2)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[1]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_3)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[2]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_4)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[3]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_5)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[4]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_6)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[5]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_7)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[6]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_8)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]),
        &i[7]);

    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_1)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[0]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_2)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[1]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_3)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[2]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_4)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[3]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_5)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[4]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_6)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[5]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_7)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[6]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_8)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[7]);
    PIKA_TEST_EQ(
        &constify(pika::util::protect(pika::util::bind(g, placeholders::_9)))(
            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]),
        &i[8]);

    // test rvalues

    PIKA_TEST_EQ(
        constify(pika::util::protect(pika::util::bind(f, placeholders::_1)))(0),
        0);

    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(0, 1),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(0, 1),
        1);

    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(0, 1, 2),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(0, 1, 2),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_3)))(0, 1, 2),
        2);

    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(0, 1, 2, 3),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(0, 1, 2, 3),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_3)))(0, 1, 2, 3),
        2);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_4)))(0, 1, 2, 3),
        3);

    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(0, 1, 2, 3, 4),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(0, 1, 2, 3, 4),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_3)))(0, 1, 2, 3, 4),
        2);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_4)))(0, 1, 2, 3, 4),
        3);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_5)))(0, 1, 2, 3, 4),
        4);

    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(0, 1, 2, 3, 4, 5),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(0, 1, 2, 3, 4, 5),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_3)))(0, 1, 2, 3, 4, 5),
        2);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_4)))(0, 1, 2, 3, 4, 5),
        3);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_5)))(0, 1, 2, 3, 4, 5),
        4);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_6)))(0, 1, 2, 3, 4, 5),
        5);

    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(0, 1, 2, 3, 4, 5, 6),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(0, 1, 2, 3, 4, 5, 6),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_3)))(0, 1, 2, 3, 4, 5, 6),
        2);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_4)))(0, 1, 2, 3, 4, 5, 6),
        3);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_5)))(0, 1, 2, 3, 4, 5, 6),
        4);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_6)))(0, 1, 2, 3, 4, 5, 6),
        5);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_7)))(0, 1, 2, 3, 4, 5, 6),
        6);

    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_1)))(0, 1, 2, 3, 4, 5, 6, 7),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_2)))(0, 1, 2, 3, 4, 5, 6, 7),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_3)))(0, 1, 2, 3, 4, 5, 6, 7),
        2);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_4)))(0, 1, 2, 3, 4, 5, 6, 7),
        3);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_5)))(0, 1, 2, 3, 4, 5, 6, 7),
        4);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_6)))(0, 1, 2, 3, 4, 5, 6, 7),
        5);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_7)))(0, 1, 2, 3, 4, 5, 6, 7),
        6);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_8)))(0, 1, 2, 3, 4, 5, 6, 7),
        7);

    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_1)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_2)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_3)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        2);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_4)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        3);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_5)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        4);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_6)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        5);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_7)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        6);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_8)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        7);
    PIKA_TEST_EQ(constify(pika::util::protect(pika::util::bind(
                    f, placeholders::_9)))(0, 1, 2, 3, 4, 5, 6, 7, 8),
        8);

    // test mixed perfect forwarding
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(i[0], 1),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(i[0], 1),
        1);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_1)))(0, i[1]),
        0);
    PIKA_TEST_EQ(constify(pika::util::protect(
                    pika::util::bind(f, placeholders::_2)))(0, i[1]),
        1);

    return pika::util::report_errors();
}
