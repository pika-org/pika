//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <pika/datastructures/optional.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threading.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <thread>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
void test_callback_register()
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
        // register another callback while callbacks are being executed
        auto f = [&] { cb2_called = true; };
        pika::stop_callback<std::function<void()>> cb2(stok, std::move(f));
    };

    pika::stop_callback<decltype(cb)> cb1(stok, cb);
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(!ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(!stok.stop_requested());
    PIKA_TEST(!cb1_called);
    PIKA_TEST(!cb2_called);

    // request stop
    auto b = ssrc.request_stop();
    PIKA_TEST(b);
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(stok.stop_requested());
    PIKA_TEST(cb1_called);
    PIKA_TEST(cb2_called);
}

///////////////////////////////////////////////////////////////////////////////
void test_callback_unregister()
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

    // register callback that unregisters itself
    bool cb1_called = false;
    pika::util::optional<pika::stop_callback<std::function<void()>>> cb;
    cb.emplace(stok, [&] {
        cb1_called = true;
        // remove this lambda in optional while being called
        cb.reset();
    });

    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(!ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(!stok.stop_requested());
    PIKA_TEST(!cb1_called);

    // request stop
    auto b = ssrc.request_stop();
    PIKA_TEST(b);
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(stok.stop_requested());
    PIKA_TEST(cb1_called);
}

///////////////////////////////////////////////////////////////////////////////
struct reg_unreg_cb
{
    pika::util::optional<pika::stop_callback<std::function<void()>>> cb{};
    bool called = false;

    void reg(pika::stop_token& stok)
    {
        cb.emplace(stok, [&] { called = true; });
    }
    void unreg()
    {
        cb.reset();
    }
};

void test_callback_concurrent_unregister()
{
    // create stop_source and stop_token:
    pika::stop_source ssrc;
    pika::stop_token stok{ssrc.get_token()};

    std::atomic<bool> cb1_called{false};
    pika::util::optional<pika::stop_callback<std::function<void()>>> opt_cb;

    auto cb1 = [&] {
        opt_cb.reset();
        cb1_called = true;
    };

    opt_cb.emplace(stok, std::ref(cb1));

    // request stop
    ssrc.request_stop();

    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(stok.stop_requested());

    PIKA_TEST(cb1_called);
}

///////////////////////////////////////////////////////////////////////////////
void test_callback_concurrent_unregister_other_thread()
{
    // create stop_source and stop_token:
    pika::stop_source ssrc;
    pika::stop_token stok{ssrc.get_token()};

    std::atomic<bool> cb1_called{false};
    pika::util::optional<pika::stop_callback<std::function<void()>>> opt_cb;

    auto cb1 = [&] {
        opt_cb.reset();
        cb1_called = true;
    };

    pika::thread t{[&] { opt_cb.emplace(stok, std::ref(cb1)); }};

    // request stop
    ssrc.request_stop();

    t.join();

    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(stok.stop_requested());

    PIKA_TEST(cb1_called);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_callback_register();
    test_callback_unregister();

    test_callback_concurrent_unregister();
    test_callback_concurrent_unregister_other_thread();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
