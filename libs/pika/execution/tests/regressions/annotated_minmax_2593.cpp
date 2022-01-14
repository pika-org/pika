//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <vector>

double compute_minmax(const std::vector<double> v)
{
    pika::execution::static_chunk_size param;
    pika::execution::parallel_task_policy par_policy;
    auto policy = par_policy.with(param);

    auto minmaxX_ = pika::minmax_element(policy, v.begin(), v.end());
    auto minmaxX = minmaxX_.get();
    return *minmaxX.max - *minmaxX.min;
}

int pika_main()
{
    std::vector<double> vec = {1.2, 3.4, 2.3, 77.8};
    double extent;

    pika::async(pika::launch::sync,
        pika::annotated_function(
            [&]() { extent = compute_minmax(vec); }, "compute_minmax"));
    PIKA_TEST_EQ(extent, 76.6);

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
