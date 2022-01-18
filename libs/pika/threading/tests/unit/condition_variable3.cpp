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
#include <pika/modules/synchronization.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threading.hpp>

#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
void test_cv_callback()
{
    bool ready{false};
    pika::lcos::local::mutex ready_mtx;
    pika::lcos::local::condition_variable_any ready_cv;

    bool cb_called{false};
    {
        pika::jthread t1{[&](pika::stop_token stoken) {
            auto f = [&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                cb_called = true;
            };
            pika::stop_callback<std::function<void()>> cb(stoken, std::move(f));

            std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
            ready_cv.wait(lg, stoken, [&ready] { return ready; });
        }};

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }    // leave scope of t1 without join() or detach() (signals cancellation)
    PIKA_TEST(cb_called);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::set_terminate([]() { PIKA_TEST(false); });
    try
    {
        test_cv_callback();
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
