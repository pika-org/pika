//  Copyright (c) 2016 David Pfander
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/iterator_support.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/datapar.hpp>

#include <cstddef>
#include <vector>

int pika_main()
{
    std::vector<double> large(64);

    auto zip_it_begin = pika::util::make_zip_iterator(large.begin());
    auto zip_it_end = pika::util::make_zip_iterator(large.end());

    pika::for_each(pika::execution::par_simd, zip_it_begin, zip_it_end,
        [](auto& t) -> void { pika::get<0>(t) = 10.0; });

    PIKA_TEST_EQ(std::count(large.begin(), large.end(), 10.0),
        std::ptrdiff_t(large.size()));

    return pika::local::finalize();    // Handles pika shutdown
}

int main(int argc, char** argv)
{
    PIKA_TEST_EQ(pika::local::init(pika_main, argc, argv), 0);
    return pika::util::report_errors();
}
