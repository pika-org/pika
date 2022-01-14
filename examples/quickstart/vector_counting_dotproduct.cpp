//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/algorithm.hpp>
#include <pika/local/init.hpp>
#include <pika/local/numeric.hpp>
#include <pika/modules/iterator_support.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    // lets say we have two vectors that simulate.. 10007D
    std::vector<double> xvalues(10007);
    std::vector<double> yvalues(10007);
    std::fill(std::begin(xvalues), std::end(xvalues), 1.0);
    std::fill(std::begin(yvalues), std::end(yvalues), 1.0);

    double result = pika::transform_reduce(pika::execution::par,
        pika::util::counting_iterator<size_t>(0),
        pika::util::counting_iterator<size_t>(10007), 0.0, std::plus<double>(),
        [&xvalues, &yvalues](size_t i) { return xvalues[i] * yvalues[i]; });
    // print the result
    std::cout << result << std::endl;

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.cfg = cfg;

    return pika::local::init(pika_main, argc, argv, init_args);
}
