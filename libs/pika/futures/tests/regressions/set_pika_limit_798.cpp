//  Copyright (c) 2013 Mario Mulansky
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #798: PIKA_LIMIT does not
// work for local dataflow

#include <pika/local/config.hpp>
#if !defined(PIKA_COMPUTE_DEVICE_CODE)
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/pack_traversal/unwrap.hpp>

// define large action
double func(double x1, double, double, double, double, double, double)
{
    return x1;
}

int pika_main()
{
    pika::shared_future<double> f = pika::make_ready_future(1.0);
    f = pika::dataflow(
        pika::launch::sync, pika::unwrapping(&func), f, f, f, f, f, f, f);
    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init(pika_main, argc, argv);
    return pika::util::report_errors();
}
#endif
