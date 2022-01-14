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

#include <atomic>
#include <chrono>
#include <string>
#include <type_traits>
#include <utility>

////////////////////////////////////////////////////////////////////////////////
void test_jthread_without_token()
{
    // test the basic jthread API (not taking stop_token arg)
    PIKA_TEST(pika::jthread::hardware_concurrency() ==
        pika::thread::hardware_concurrency());

    pika::stop_token stoken;
    PIKA_TEST(!stoken.stop_possible());

    {
        pika::jthread::id id{pika::this_thread::get_id()};
        std::atomic<bool> all_set{false};

        pika::jthread t([&id, &all_set] {    // NOTE: no stop_token passed
            // check some values of the started thread
            id = pika::this_thread::get_id();
            all_set.store(true);

            // wait until loop is done (no interrupt checked)
            for (int c = 9; c >= 0; --c)
            {
                pika::this_thread::yield();
            }
        });

        // wait until t has set all initial values
        for (int i = 0; !all_set.load(); ++i)
        {
            pika::this_thread::yield();
        }

        // and check all values
        PIKA_TEST(t.joinable());
        PIKA_TEST(id == t.get_id());
        stoken = t.get_stop_token();
        PIKA_TEST(!stoken.stop_requested());
    }    // leave scope of t without join() or detach() (signals cancellation)

    PIKA_TEST(stoken.stop_requested());
}

////////////////////////////////////////////////////////////////////////////////
void test_jthread_with_token()
{
    pika::stop_source ssource;
    pika::stop_source origsource;
    PIKA_TEST(ssource.stop_possible());
    PIKA_TEST(!ssource.stop_requested());

    {
        pika::jthread::id id{pika::this_thread::get_id()};
        std::atomic<bool> all_set{false};
        std::atomic<bool> done{false};
        pika::jthread t(
            [&id, &all_set, &done](pika::stop_token stoptoken) {
                // check some values of the started thread
                id = pika::this_thread::get_id();
                all_set.store(true);

                // wait until interrupt is signaled
                for (int i = 0; !stoptoken.stop_requested(); ++i)
                {
                    pika::this_thread::yield();
                }

                done.store(true);
            },
            ssource.get_token());

        // wait until t has set all initial values
        for (int i = 0; !all_set.load(); ++i)
        {
            pika::this_thread::yield();
        }

        // and check all values
        PIKA_TEST(t.joinable());
        PIKA_TEST(id == t.get_id());

        pika::this_thread::yield();

        origsource = std::move(ssource);
        ssource = t.get_stop_source();
        PIKA_TEST(!ssource.stop_requested());

        auto ret = ssource.request_stop();
        PIKA_TEST(ret);

        ret = ssource.request_stop();
        PIKA_TEST(!ret);
        PIKA_TEST(ssource.stop_requested());
        PIKA_TEST(!done.load());
        PIKA_TEST(!origsource.stop_requested());

        pika::this_thread::yield();
        origsource.request_stop();
    }    // leave scope of t without join() or detach() (signals cancellation)

    PIKA_TEST(origsource.stop_requested());
    PIKA_TEST(ssource.stop_requested());
}

////////////////////////////////////////////////////////////////////////////////
void test_join()
{
    // test jthread join()
    pika::stop_source ssource;
    PIKA_TEST(ssource.stop_possible());

    {
        pika::jthread t([](pika::stop_token stoken) {
            // wait until interrupt is signaled (due to calling request_stop()
            // for the token)
            for (int i = 0; !stoken.stop_requested(); ++i)
            {
                pika::this_thread::yield();
            }
        });
        ssource = t.get_stop_source();

        // let another thread signal cancellation after some time
        pika::jthread t2([ssource]() mutable {
            // just wait for a while
            pika::this_thread::yield();

            // signal interrupt to other thread
            ssource.request_stop();
        });

        // wait for all thread to finish
        t2.join();
        PIKA_TEST(!t2.joinable());
        PIKA_TEST(t.joinable());

        t.join();
        PIKA_TEST(!t.joinable());
    }    // leave scope of t without join() or detach() (signals cancellation)
}

////////////////////////////////////////////////////////////////////////////////
void test_detach()
{
    // test jthread detach()
    pika::stop_source ssource;
    PIKA_TEST(ssource.stop_possible());
    std::atomic<bool> finally_interrupted{false};

    {
        pika::jthread t0;
        pika::jthread::id id{pika::this_thread::get_id()};
        bool is_interrupted;
        pika::stop_token interrupt_token;
        std::atomic<bool> all_set{false};

        pika::jthread t([&id, &is_interrupted, &interrupt_token, &all_set,
                           &finally_interrupted](pika::stop_token stoken) {
            // check some values of the started thread
            id = pika::this_thread::get_id();
            interrupt_token = stoken;
            is_interrupted = stoken.stop_requested();
            PIKA_TEST(stoken.stop_possible());
            PIKA_TEST(!is_interrupted);
            all_set.store(true);

            // wait until interrupt is signaled (due to calling request_stop()
            // for the token)
            for (int i = 0; !stoken.stop_requested(); ++i)
            {
                pika::this_thread::yield();
            }

            finally_interrupted.store(true);
        });

        // wait until t has set all initial values
        for (int i = 0; !all_set.load(); ++i)
        {
            pika::this_thread::yield();
        }

        // and check all values
        PIKA_TEST(!t0.joinable());
        PIKA_TEST(t.joinable());
        PIKA_TEST(id == t.get_id());
        PIKA_TEST(!is_interrupted);
        PIKA_TEST(interrupt_token == t.get_stop_source().get_token());

        ssource = t.get_stop_source();
        PIKA_TEST(interrupt_token.stop_possible());
        PIKA_TEST(!interrupt_token.stop_requested());

        t.detach();
        PIKA_TEST(!t.joinable());
    }    // leave scope of t without join() or detach()

    // finally signal cancellation
    PIKA_TEST(!finally_interrupted.load());
    ssource.request_stop();

    // and check consequences
    PIKA_TEST(ssource.stop_requested());
    for (int i = 0; !finally_interrupted.load() && i < 100; ++i)
    {
        pika::this_thread::yield();
    }

    PIKA_TEST(finally_interrupted.load());
}

////////////////////////////////////////////////////////////////////////////////
void test_pika_thread()
{
    // test the extended pika::thread API
    pika::thread t0;
    pika::thread::id id{pika::this_thread::get_id()};
    std::atomic<bool> all_set{false};
    pika::stop_source shall_die;
    pika::thread t([&id, &all_set, shall_die = shall_die.get_token()] {
        // check some supplementary values of the started thread
        id = pika::this_thread::get_id();
        all_set.store(true);

        // and wait until cancellation is signaled via passed token
        bool caught_exception = false;
        try
        {
            for (int i = 0;; ++i)
            {
                if (shall_die.stop_requested())
                {
                    throw "interrupted";
                }
                pika::this_thread::yield();
            }
            PIKA_TEST(false);
        }
        catch (std::exception&)
        {
            // "interrupted" not derived from std::exception
            PIKA_TEST(false);
        }
        catch (const char* e)
        {
            caught_exception = true;
        }
        catch (...)
        {
            PIKA_TEST(false);
        }
        PIKA_TEST(caught_exception);
        PIKA_TEST(shall_die.stop_requested());
    });

    // wait until t has set all initial values
    for (int i = 0; !all_set.load(); ++i)
    {
        pika::this_thread::yield();
    }

    // and check all values
    PIKA_TEST(id == t.get_id());

    // signal cancellation via manually installed interrupt token
    shall_die.request_stop();
    t.join();
}

//////////////////////////////////////////////////////////////////////////////
void test_temporarily_disable_token()
{
    // test exchanging the token to disable it temporarily
    enum class State
    {
        init,
        loop,
        disabled,
        restored,
        interrupted
    };

    std::atomic<State> state{State::init};
    pika::stop_source tis;

    {
        pika::jthread t([&state](pika::stop_token stoken) {
            auto actToken = stoken;

            // just loop (no interrupt should occur)
            state.store(State::loop);
            try
            {
                for (int i = 0; i < 10; ++i)
                {
                    if (actToken.stop_requested())
                    {
                        throw "interrupted";
                    }
                    pika::this_thread::yield();
                }
            }
            catch (...)
            {
                PIKA_TEST(false);
            }

            // temporarily disable interrupts
            pika::stop_token interrupt_disabled;
            std::swap(stoken, interrupt_disabled);
            state.store(State::disabled);

            // loop again until interrupt signaled to original interrupt token
            try
            {
                while (!actToken.stop_requested())
                {
                    if (stoken.stop_requested())
                    {
                        throw "interrupted";
                    }
                    pika::this_thread::yield();
                }

                for (int i = 0; i < 10; ++i)
                {
                    pika::this_thread::yield();
                }
            }
            catch (...)
            {
                PIKA_TEST(false);
            }
            state.store(State::restored);

            // enable interrupts again
            std::swap(stoken, interrupt_disabled);

            // loop again (should immediately throw)
            PIKA_TEST(!interrupt_disabled.stop_requested());
            try
            {
                if (actToken.stop_requested())
                {
                    throw "interrupted";
                }
            }
            catch (const char*)
            {
                state.store(State::interrupted);
            }
        });

        while (state.load() != State::disabled)
        {
            pika::this_thread::yield();
        }

        pika::this_thread::yield();
        tis = t.get_stop_source();
    }    // leave scope of t without join() or detach() (signals cancellation)

    PIKA_TEST(tis.stop_requested());
    PIKA_TEST(state.load() == State::interrupted);
}

///////////////////////////////////////////////////////////////////////////////
void test_jthread_api()
{
    PIKA_TEST(pika::jthread::hardware_concurrency() ==
        pika::thread::hardware_concurrency());

    pika::stop_source ssource;
    PIKA_TEST(ssource.stop_possible());
    PIKA_TEST(ssource.get_token().stop_possible());

    pika::stop_token stoken;
    PIKA_TEST(!stoken.stop_possible());

    // thread with no callable and invalid source
    pika::jthread t0;
    pika::jthread::native_handle_type nh = t0.native_handle();
    PIKA_TEST(
        (std::is_same<decltype(nh), pika::thread::native_handle_type>::value));
    PIKA_TEST(!t0.joinable());

    pika::stop_source ssourceStolen{std::move(ssource)};
    // NOLINTNEXTLINE(bugprone-use-after-move)
    PIKA_TEST(!ssource.stop_possible());
    PIKA_TEST(ssource == t0.get_stop_source());
    PIKA_TEST(ssource.get_token() == t0.get_stop_token());

    {
        pika::jthread::id id{pika::this_thread::get_id()};
        pika::stop_token interrupt_token;
        std::atomic<bool> all_set{false};
        pika::jthread t(
            [&id, &interrupt_token, &all_set](pika::stop_token stoken) {
                // check some values of the started thread
                id = pika::this_thread::get_id();
                interrupt_token = stoken;
                PIKA_TEST(stoken.stop_possible());
                PIKA_TEST(!stoken.stop_requested());
                all_set.store(true);

                // wait until interrupt is signaled (due to destructor of t)
                for (int i = 0; !stoken.stop_requested(); ++i)
                {
                    pika::this_thread::yield();
                }
            });

        // wait until t has set all initial values
        for (int i = 0; !all_set.load(); ++i)
        {
            pika::this_thread::yield();
        }

        // and check all values
        PIKA_TEST(t.joinable());
        PIKA_TEST(id == t.get_id());
        PIKA_TEST(interrupt_token == t.get_stop_source().get_token());
        PIKA_TEST(interrupt_token == t.get_stop_token());
        stoken = t.get_stop_source().get_token();
        stoken = t.get_stop_token();
        PIKA_TEST(interrupt_token.stop_possible());
        PIKA_TEST(!interrupt_token.stop_requested());

        // test swap()
        std::swap(t0, t);
        PIKA_TEST(!t.joinable());
        PIKA_TEST(pika::stop_token{} == t.get_stop_source().get_token());
        PIKA_TEST(pika::stop_token{} == t.get_stop_token());
        PIKA_TEST(t0.joinable());
        PIKA_TEST(id == t0.get_id());
        PIKA_TEST(interrupt_token == t0.get_stop_source().get_token());
        PIKA_TEST(interrupt_token == t0.get_stop_token());

        // manual swap with move()
        auto ttmp{std::move(t0)};
        t0 = std::move(t);
        t = std::move(ttmp);
        PIKA_TEST(!t0.joinable());
        PIKA_TEST(pika::stop_token{} == t0.get_stop_source().get_token());
        PIKA_TEST(pika::stop_token{} == t0.get_stop_token());
        PIKA_TEST(t.joinable());
        PIKA_TEST(id == t.get_id());
        PIKA_TEST(interrupt_token == t.get_stop_source().get_token());
        PIKA_TEST(interrupt_token == t.get_stop_token());
    }    // leave scope of t without join() or detach() (signals cancellation)

    PIKA_TEST(stoken.stop_requested());
}

////////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::set_terminate([]() { PIKA_TEST(false); });

    test_jthread_without_token();
    test_jthread_with_token();
    test_join();
    test_detach();
    test_pika_thread();
    test_temporarily_disable_token();
    test_jthread_api();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
