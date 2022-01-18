//  Copyright (c) 2020 Steven R. Brandt
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #5016: pika::parallel::fill fails compiling

// suppress deprecation warnings for algorithms
#define PIKA_HAVE_DEPRECATION_WARNINGS_V0_1 0

#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/fill.hpp>

#include <algorithm>
#include <vector>

void fill_example()
{
    pika::execution::parallel_executor exec;

    std::vector<float> vd(5);
    pika::parallel::fill(
        pika::execution::par.on(exec), vd.begin(), vd.end(), 2.0f);

    std::vector<float> vd1(5);
    pika::fill(pika::execution::par.on(exec), vd1.begin(), vd1.end(), 2.0f);

    std::vector<float> expected(5);
    std::fill(expected.begin(), expected.end(), 2.0f);

    PIKA_TEST(vd == expected);
    PIKA_TEST(vd1 == expected);
}

int pika_main()
{
    fill_example();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
