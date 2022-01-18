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

#include <chrono>
#include <utility>

void test_stop_token_basic_api()
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
    bool cb1called{false};
    auto cb1 = [&] { cb1called = true; };
    {
        pika::stop_callback<decltype(cb1)> scb1(stok, cb1);    // copies cb1
        PIKA_TEST(ssrc.stop_possible());
        PIKA_TEST(!ssrc.stop_requested());
        PIKA_TEST(stok.stop_possible());
        PIKA_TEST(!stok.stop_requested());
        PIKA_TEST(!cb1called);
    }    // unregister callback

    // register another callback
    bool cb2called{false};
    auto cb2 = [&] {
        PIKA_TEST(stok.stop_requested());
        cb2called = true;
    };

    pika::stop_callback<decltype(cb2)> scb2a{stok, cb2};    // copies cb2
    pika::stop_callback<decltype(cb2)> scb2b{stok, std::move(cb2)};
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(!ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(!stok.stop_requested());
    PIKA_TEST(!cb1called);
    PIKA_TEST(!cb2called);

    // request stop
    auto b = ssrc.request_stop();
    PIKA_TEST(b);
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(stok.stop_requested());
    PIKA_TEST(!cb1called);
    PIKA_TEST(cb2called);

    b = ssrc.request_stop();
    PIKA_TEST(!b);

    // register another callback
    bool cb3called{false};

    auto cb3 = [&] { cb3called = true; };
    pika::stop_callback<decltype(cb3)> scb3(stok, std::move(cb3));
    PIKA_TEST(ssrc.stop_possible());
    PIKA_TEST(ssrc.stop_requested());
    PIKA_TEST(stok.stop_possible());
    PIKA_TEST(stok.stop_requested());
    PIKA_TEST(!cb1called);
    PIKA_TEST(cb2called);
    PIKA_TEST(cb3called);
}

///////////////////////////////////////////////////////////////////////////////
void test_stop_token_api()
{
    // stop_source: create, copy, assign and destroy
    {
        pika::stop_source is1;
        pika::stop_source is2{is1};
        pika::stop_source is3 = is1;
        pika::stop_source is4{std::move(is1)};
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(!is1.stop_possible());
        PIKA_TEST(is2.stop_possible());
        PIKA_TEST(is3.stop_possible());
        PIKA_TEST(is4.stop_possible());
        is1 = is2;
        PIKA_TEST(is1.stop_possible());
        is1 = std::move(is2);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(!is2.stop_possible());
        std::swap(is1, is2);
        PIKA_TEST(!is1.stop_possible());
        PIKA_TEST(is2.stop_possible());
        is1.swap(is2);
        PIKA_TEST(is1.stop_possible());
        PIKA_TEST(!is2.stop_possible());

        // stop_source without shared stop state
        pika::stop_source is0{pika::nostopstate};
        PIKA_TEST(!is0.stop_requested());
        PIKA_TEST(!is0.stop_possible());
    }

    // stop_token: create, copy, assign and destroy
    {
        pika::stop_token it1;
        pika::stop_token it2{it1};
        pika::stop_token it3 = it1;
        pika::stop_token it4{std::move(it1)};
        it1 = it2;
        it1 = std::move(it2);
        std::swap(it1, it2);
        it1.swap(it2);
        PIKA_TEST(!it1.stop_possible());
        PIKA_TEST(!it2.stop_possible());
        PIKA_TEST(!it3.stop_possible());
        PIKA_TEST(!it4.stop_possible());
    }

    // tokens without an source are no longer interruptible
    {
        pika::stop_source* isp = new pika::stop_source;
        pika::stop_source& isr = *isp;
        pika::stop_token it{isr.get_token()};
        PIKA_TEST(isr.stop_possible());
        PIKA_TEST(it.stop_possible());
        delete isp;    // not interrupted and losing last source
        PIKA_TEST(!it.stop_possible());
    }

    {
        pika::stop_source* isp = new pika::stop_source;
        pika::stop_source& isr = *isp;
        pika::stop_token it{isr.get_token()};
        PIKA_TEST(isr.stop_possible());
        PIKA_TEST(it.stop_possible());
        isr.request_stop();
        delete isp;    // interrupted and losing last source
        PIKA_TEST(it.stop_possible());
    }

    // stop_possible(), stop_requested(), and request_stop()
    {
        pika::stop_source is_not_valid;
        pika::stop_source is_not_stopped{std::move(is_not_valid)};
        pika::stop_source is_stopped;
        is_stopped.request_stop();

        // NOLINTNEXTLINE(bugprone-use-after-move)
        pika::stop_token it_not_valid{is_not_valid.get_token()};
        pika::stop_token it_not_stopped{is_not_stopped.get_token()};
        pika::stop_token it_stopped{is_stopped.get_token()};

        // stop_possible() and stop_requested()
        PIKA_TEST(!is_not_valid.stop_possible());
        PIKA_TEST(is_not_stopped.stop_possible());
        PIKA_TEST(is_stopped.stop_possible());
        PIKA_TEST(!is_not_valid.stop_requested());
        PIKA_TEST(!is_not_stopped.stop_requested());
        PIKA_TEST(is_stopped.stop_requested());

        // stop_possible() and stop_requested()
        PIKA_TEST(!it_not_valid.stop_possible());
        PIKA_TEST(it_not_stopped.stop_possible());
        PIKA_TEST(it_stopped.stop_possible());
        PIKA_TEST(!it_not_stopped.stop_requested());
        PIKA_TEST(it_stopped.stop_requested());

        // request_stop()
        PIKA_TEST(is_not_stopped.request_stop() == true);
        PIKA_TEST(is_not_stopped.request_stop() == false);
        PIKA_TEST(is_stopped.request_stop() == false);
        PIKA_TEST(is_not_stopped.stop_requested());
        PIKA_TEST(is_stopped.stop_requested());
        PIKA_TEST(it_not_stopped.stop_requested());
        PIKA_TEST(it_stopped.stop_requested());
    }

    // assignment and swap()
    {
        pika::stop_source is_not_valid;
        pika::stop_source is_not_stopped{std::move(is_not_valid)};
        pika::stop_source is_stopped;
        is_stopped.request_stop();

        // NOLINTNEXTLINE(bugprone-use-after-move)
        pika::stop_token it_not_valid{is_not_valid.get_token()};
        pika::stop_token it_not_stopped{is_not_stopped.get_token()};
        pika::stop_token it_stopped{is_stopped.get_token()};

        // assignments and swap()
        PIKA_TEST(!pika::stop_token{}.stop_requested());
        it_stopped = pika::stop_token{};
        PIKA_TEST(!it_stopped.stop_possible());
        PIKA_TEST(!it_stopped.stop_requested());
        is_stopped = pika::stop_source{};
        PIKA_TEST(is_stopped.stop_possible());
        PIKA_TEST(!is_stopped.stop_requested());

        std::swap(it_stopped, it_not_valid);
        PIKA_TEST(!it_stopped.stop_possible());
        PIKA_TEST(!it_not_valid.stop_possible());
        PIKA_TEST(!it_not_valid.stop_requested());
        pika::stop_token itnew = std::move(it_not_valid);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(!it_not_valid.stop_possible());

        std::swap(is_stopped, is_not_valid);
        PIKA_TEST(!is_stopped.stop_possible());
        PIKA_TEST(is_not_valid.stop_possible());
        PIKA_TEST(!is_not_valid.stop_requested());
        pika::stop_source isnew = std::move(is_not_valid);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(!is_not_valid.stop_possible());
    }

    // shared ownership semantics
    pika::stop_source is;
    pika::stop_token it1{is.get_token()};
    pika::stop_token it2{it1};
    PIKA_TEST(is.stop_possible() && !is.stop_requested());
    PIKA_TEST(it1.stop_possible() && !it1.stop_requested());
    PIKA_TEST(it2.stop_possible() && !it2.stop_requested());
    is.request_stop();
    PIKA_TEST(is.stop_possible() && is.stop_requested());
    PIKA_TEST(it1.stop_possible() && it1.stop_requested());
    PIKA_TEST(it2.stop_possible() && it2.stop_requested());

    // == and !=
    {
        pika::stop_source is_not_valid1;
        pika::stop_source is_not_valid2;
        pika::stop_source is_not_stopped1{std::move(is_not_valid1)};
        // NOLINTNEXTLINE(bugprone-use-after-move)
        pika::stop_source is_not_stopped2{is_not_stopped1};
        pika::stop_source is_stopped1{std::move(is_not_valid2)};
        // NOLINTNEXTLINE(bugprone-use-after-move)
        pika::stop_source is_stopped2{is_stopped1};
        is_stopped1.request_stop();

        // NOLINTNEXTLINE(bugprone-use-after-move)
        pika::stop_token it_not_valid1{is_not_valid1.get_token()};
        // NOLINTNEXTLINE(bugprone-use-after-move)
        pika::stop_token it_not_valid2{is_not_valid2.get_token()};
        pika::stop_token it_not_valid3;
        pika::stop_token it_not_stopped1{is_not_stopped1.get_token()};
        pika::stop_token it_not_stopped2{is_not_stopped2.get_token()};
        pika::stop_token it_not_stopped3{it_not_stopped1};
        pika::stop_token it_stopped1{is_stopped1.get_token()};
        pika::stop_token it_stopped2{is_stopped2.get_token()};
        pika::stop_token it_stopped3{it_stopped2};

        PIKA_TEST(is_not_valid1 == is_not_valid2);
        PIKA_TEST(is_not_stopped1 == is_not_stopped2);
        PIKA_TEST(is_stopped1 == is_stopped2);
        PIKA_TEST(is_not_valid1 != is_not_stopped1);
        PIKA_TEST(is_not_valid1 != is_stopped1);
        PIKA_TEST(is_not_stopped1 != is_stopped1);

        PIKA_TEST(it_not_valid1 == it_not_valid2);
        PIKA_TEST(it_not_valid2 == it_not_valid3);
        PIKA_TEST(it_not_stopped1 == it_not_stopped2);
        PIKA_TEST(it_not_stopped2 == it_not_stopped3);
        PIKA_TEST(it_stopped1 == it_stopped2);
        PIKA_TEST(it_stopped2 == it_stopped3);
        PIKA_TEST(it_not_valid1 != it_not_stopped1);
        PIKA_TEST(it_not_valid1 != it_stopped1);
        PIKA_TEST(it_not_stopped1 != it_stopped1);

        PIKA_TEST(!(is_not_valid1 != is_not_valid2));
        PIKA_TEST(!(it_not_valid1 != it_not_valid2));
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename D>
void sleep(D dur)
{
    if (dur > std::chrono::milliseconds{0})
    {
        std::this_thread::sleep_for(dur);
    }
}

template <typename D>
void test_stoken(D dur)
{
    int okSteps = 0;
    try
    {
        pika::stop_token it0;    // should not allocate anything

        pika::stop_source interruptor;
        pika::stop_token interruptee{interruptor.get_token()};
        ++okSteps;
        sleep(dur);    // 1
        PIKA_TEST(!interruptor.stop_requested());
        PIKA_TEST(!interruptee.stop_requested());

        interruptor.request_stop();    // INTERRUPT !!!
        ++okSteps;
        sleep(dur);    // 2
        PIKA_TEST(interruptor.stop_requested());
        PIKA_TEST(interruptee.stop_requested());

        interruptor.request_stop();
        ++okSteps;
        sleep(dur);    // 3
        PIKA_TEST(interruptor.stop_requested());
        PIKA_TEST(interruptee.stop_requested());

        interruptor = pika::stop_source{};
        interruptee = interruptor.get_token();
        ++okSteps;
        sleep(dur);    // 4
        PIKA_TEST(!interruptor.stop_requested());
        PIKA_TEST(!interruptee.stop_requested());

        interruptor.request_stop();    // INTERRUPT !!!
        ++okSteps;
        sleep(dur);    // 5
        PIKA_TEST(interruptor.stop_requested());
        PIKA_TEST(interruptee.stop_requested());

        interruptor.request_stop();
        ++okSteps;
        sleep(dur);    // 6
        PIKA_TEST(interruptor.stop_requested());
        PIKA_TEST(interruptee.stop_requested());
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
    PIKA_TEST(okSteps == 6);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    test_stop_token_basic_api();
    test_stop_token_api();
    test_stoken(std::chrono::seconds{0});
    test_stoken(std::chrono::milliseconds{500});

    pika::local::finalize();
    return pika::util::report_errors();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
