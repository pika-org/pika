//  Copyright 2015 (c) Dominic Marcello
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1751:
// pika::future::wait_for fails a simple test

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

#include <chrono>
#include <iostream>

int pika_main()
{
    auto overall_start_time = std::chrono::high_resolution_clock::now();

    while (true)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // run for 3 seconds max
        std::chrono::duration<double> overall_dif =
            start_time - overall_start_time;
        if (overall_dif.count() > 3.0)
            break;

        auto f = pika::async([]() {});

        if (f.wait_for(std::chrono::seconds(1)) == pika::future_status::timeout)
        {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dif = now - start_time;

            PIKA_TEST_LTE(dif.count(), 1.1);
            break;
        }
        else
        {
            f.get();
        }

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dif = now - start_time;
        PIKA_TEST_LT(dif.count(), 1.1);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    pika::local::init(pika_main, argc, argv);
    return pika::util::report_errors();
}
