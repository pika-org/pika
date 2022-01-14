//  Copyright (c) 2020 LiliumAtratum
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// see #4786: transform_inclusive_scan tries to implicitly convert between
//            types, instead of using the provided `conv` function

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/transform_inclusive_scan.hpp>

#include <vector>

struct Integer
{
    int integer;
    Integer()
      : integer(0)
    {
    }
    explicit Integer(int i)
      : integer(i)
    {
    }
};

bool operator==(Integer lhs, Integer rhs)
{
    return lhs.integer == rhs.integer;
}

int pika_main()
{
    std::vector<int> test{1, 10, 100, 1000};
    std::vector<Integer> output(test.size());

    pika::transform_inclusive_scan(
        pika::execution::par, test.cbegin(), test.cend(), output.begin(),
        [](Integer acc, Integer xs) -> Integer {
            return Integer{acc.integer + xs.integer};
        },
        [](int el) -> Integer { return Integer{el}; });

    std::vector<Integer> expected = {
        Integer{1}, Integer{11}, Integer{111}, Integer{1111}};
    PIKA_TEST(output == expected);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
