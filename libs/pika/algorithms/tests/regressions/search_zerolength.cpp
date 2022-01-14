//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/search.hpp>

#include <numeric>
#include <string>
#include <vector>

void search_zero_dist_test()
{
    using pika::execution::par;
    using pika::execution::seq;
    using pika::execution::task;

    typedef std::vector<int>::iterator iterator;

    std::vector<int> c(10007);
    std::iota(c.begin(), c.end(), 1);
    std::vector<int> h(0);

    pika::future<iterator> fut_seq =
        pika::search(seq(task), c.begin(), c.end(), h.begin(), h.end());
    pika::future<iterator> fut_par =
        pika::search(par(task), c.begin(), c.end(), h.begin(), h.end());

    PIKA_TEST(fut_seq.get() == c.begin());
    PIKA_TEST(fut_par.get() == c.begin());
}

int pika_main()
{
    search_zero_dist_test();
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    pika::local::init_params init_args;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exted with non-zero status");

    return pika::util::report_errors();
}
