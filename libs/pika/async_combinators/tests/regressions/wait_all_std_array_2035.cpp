//  Copyright (c) 2016 Hadrian G. (a.k.a. Neolander)
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This compile-only test case verifies that #2035 remains fixed

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <array>

int pika_main()
{
    std::array<pika::shared_future<int>, 1> future_array{
        {pika::make_ready_future(0)}};

    pika::wait_all(future_array.cbegin(), future_array.cend());

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
