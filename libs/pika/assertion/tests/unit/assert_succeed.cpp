//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/testing.hpp>

import pika;
import pika.testing;

int main()
{
    PIKA_ASSERT(true);

    // This test should just run without crashing
    PIKA_TEST(true);
}
