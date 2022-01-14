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

void is_sentinel_for()
{
    PIKA_TEST_MSG((pika::traits::is_sentinel_for<sentinel<int64_t>,
                     iterator<std::int64_t>>::value),
        "Sentinel value is not proper for given iterator");

    PIKA_TEST_MSG(
        (!pika::traits::is_sentinel_for<std::int64_t, std::int64_t>::value),
        "Integer - integer pair is incompatible pair");

    PIKA_TEST_MSG((pika::traits::is_sentinel_for<std::vector<int>::iterator,
                     std::vector<int>::iterator>::value),
        "Incompatible begin - end iterator pair on vector");

    PIKA_TEST_MSG((!pika::traits::is_sentinel_for<std::string,
                     std::string::iterator>::value),
        "String - string::iterator is incompatible pair");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        is_sentinel_for();
    }

    return pika::util::report_errors();
}
