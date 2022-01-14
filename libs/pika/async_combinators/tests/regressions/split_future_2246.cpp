//  Copyright (c) 2016 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    using pair_type = std::pair<std::map<int, int>, std::set<int>>;
    using pair_fut_type =
        std::pair<pika::future<std::map<int, int>>, pika::future<std::set<int>>>;

    std::map<int, int> mm;
    // fill mm with arbitrary values
    mm[123] = 321;
    mm[999] = 999;
    mm[6] = 43556;

    // fill ss with arbitrary values
    std::set<int> ss{0, 1, 10, 100, 101, 2000};

    pair_type p = std::make_pair(mm, ss);
    pika::future<pair_type> f_pair = pika::make_ready_future(p);

    // given a future of a pair, get a pair of futures
    pair_fut_type pair_f = pika::split_future(std::move(f_pair));

    // see if the values of mm2 and ss2 are the same
    std::map<int, int> mm2 = pair_f.first.get();
    std::cout << "Printing map: ";
    for (auto val : mm2)
        std::cout << "(" << val.first << ", " << val.second << ") ";
    std::cout << std::endl;

    PIKA_TEST_EQ(mm.size(), mm2.size());
    std::map<int, int>::const_iterator mm_it = mm.begin(), mm_it2 = mm2.begin();
    std::map<int, int>::const_iterator mm_end = mm.end(), mm_end2 = mm2.end();
    for (/**/; mm_it != mm_end && mm_it2 != mm_end2; ++mm_it, ++mm_it2)
    {
        PIKA_TEST_EQ((*mm_it).first, (*mm_it2).first);
        PIKA_TEST_EQ((*mm_it).second, (*mm_it2).second);
    }

    std::set<int> ss2 = pair_f.second.get();
    std::cout << "Printing set: ";
    for (auto val : ss2)
        std::cout << val << " ";
    std::cout << std::endl;

    PIKA_TEST_EQ(ss.size(), ss2.size());
    std::set<int>::const_iterator ss_it = ss.begin(), ss_it2 = ss2.begin();
    std::set<int>::const_iterator ss_end = ss.end(), ss_end2 = ss2.end();
    for (/**/; ss_it != ss_end && ss_it2 != ss_end2; ++ss_it, ++ss_it2)
    {
        PIKA_TEST_EQ(*ss_it, *ss_it2);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv), 0);
    return pika::util::report_errors();
}
