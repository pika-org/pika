//  Copyright (c) 2016 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/fill.hpp>

int main()
{
    const int size = 1000000;
    float* a = new float[size];

    bool caught_exception = false;
    try
    {
        // this should throw as the pika runtime has not been initialized
        pika::fill(pika::execution::par, a, a + size, 1.0f);

        // fill should have thrown
        PIKA_TEST(false);
    }
    catch (pika::exception const&)
    {
        caught_exception = true;
    }

    PIKA_TEST(caught_exception);

    delete[] a;

    return 0;
}
