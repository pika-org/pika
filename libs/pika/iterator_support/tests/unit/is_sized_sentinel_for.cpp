//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/iterator_support/traits/is_sentinel_for.hpp>
#include <pika/modules/testing.hpp>

#include <cstdint>
#include <string>
#include <vector>

void is_sized_sentinel_for()
{
    PIKA_TEST_MSG((pika::traits::is_sized_sentinel_for<sentinel<int64_t>,
                      iterator<std::int64_t>>::value == false),
        "Sentinel falsely marked as sized for particular iterator");

    PIKA_TEST_MSG((pika::traits::is_sized_sentinel_for<std::vector<int>::iterator,
                      std::vector<int>::iterator>::value == true),
        "Begin end iterator (vector) pair should be sized");

    PIKA_TEST_MSG((pika::traits::is_sized_sentinel_for<std::string::iterator,
                      std::string::iterator>::value == true),
        "Begin end iterator (string) pair should be sized");

    PIKA_TEST_MSG((pika::traits::is_sized_sentinel_for<std::int64_t,
                      std::vector<int>::iterator>::value == false),
        "Integer is not a sized sentinel for a vector iterator");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        is_sized_sentinel_for();
    }

    return pika::util::report_errors();
}
