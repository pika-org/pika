//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3641: Trouble with using ranges-v3 and pika::parallel::reduce
// #3646: Parallel algorithms should accept iterator/sentinel pairs

#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/reduce.hpp>

#include <cstdint>

int pika_main()
{
    std::int64_t result = pika::ranges::reduce(pika::execution::seq,
        iterator<std::int64_t>{0}, sentinel<int64_t>{100}, std::int64_t(0));

    PIKA_TEST_EQ(result, std::int64_t(4950));

    result = pika::ranges::reduce(pika::execution::par, iterator<std::int64_t>{0},
        sentinel<int64_t>{100}, std::int64_t(0));

    PIKA_TEST_EQ(result, std::int64_t(4950));

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
