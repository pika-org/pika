//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threading.hpp>

#include <iostream>

///////////////////////////////////////////////////////////////////////////////
void test_callback_throw()
{
    // create stop_source
    pika::stop_source ssrc;
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(!ssrc.stop_requested());

    // create stop_token from stop_source
    pika::stop_token stok{ssrc.get_token()};
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(!ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(!stok.stop_requested());

    // register callback
    bool cb1_called{false};
    bool cb2_called{false};

    auto cb = [&] {
        cb1_called = true;
        // throw
        throw "callback called";
    };

    pika::stop_callback<decltype(cb)> cb1(stok, cb);

    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(!ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(!stok.stop_requested());
    PIKA_TEST(!cb1_called);
    PIKA_TEST(!cb2_called);

    // catch terminate() call:
    std::set_terminate([] {
        std::cout << "std::terminate called\n";
        std::exit(pika::util::report_errors());
    });

    // request stop
    ssrc.request_stop();
    PIKA_TEST(false);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    // this test terminates execution
    test_callback_throw();

    return pika::util::report_errors();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
