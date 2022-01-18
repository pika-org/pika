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
#include <thread>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
void test_interrupt_by_destructor()
{
    auto interval = std::chrono::milliseconds(200);
    bool was_interrupted = false;

    {
        pika::jthread t([interval, &was_interrupted](pika::stop_token stoken) {
            PIKA_TEST(!stoken.stop_requested());
            try
            {
                // loop until interrupted (at most 40 times the interval)
                for (int i = 0; i < 40; ++i)
                {
                    if (stoken.stop_requested())
                    {
                        throw "interrupted";
                    }

                    std::this_thread::sleep_for(interval);
                    pika::this_thread::yield();
                }
                PIKA_TEST(false);
            }
            catch (std::exception&)
            {
                // "interrupted" not derived from std::exception
                PIKA_TEST(false);
            }
            catch (const char*)
            {
                PIKA_TEST(stoken.stop_requested());
                was_interrupted = true;
            }
            catch (...)
            {
                PIKA_TEST(false);
            }
        });

        PIKA_TEST(!t.get_stop_source().stop_requested());

        // call destructor after 4 times the interval (should signal the interrupt)
        std::this_thread::sleep_for(4 * interval);
        PIKA_TEST(!t.get_stop_source().stop_requested());
    }

    // key assertion: signaled interrupt was processed
    PIKA_TEST(was_interrupted);
}

///////////////////////////////////////////////////////////////////////////////
void test_interrupt_started_thread()
{
    auto interval = std::chrono::milliseconds(200);

    {
        bool interrupted = false;
        pika::jthread t([interval, &interrupted](pika::stop_token stoken) {
            try
            {
                // loop until interrupted (at most 40 times the interval)
                for (int i = 0; i < 40; ++i)
                {
                    if (stoken.stop_requested())
                    {
                        throw "interrupted";
                    }
                    std::this_thread::sleep_for(interval);
                }
                PIKA_TEST(false);
            }
            catch (...)
            {
                interrupted = true;
            }
        });

        std::this_thread::sleep_for(4 * interval);
        t.request_stop();
        PIKA_TEST(t.get_stop_source().stop_requested());
        t.join();
        PIKA_TEST(interrupted);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_interrupt_started_thread_with_subthread()
{
    auto interval = std::chrono::milliseconds(200);

    {
        pika::jthread t([interval](pika::stop_token stoken) {
            pika::jthread t2([interval, stoken] {
                while (!stoken.stop_requested())
                {
                    std::this_thread::sleep_for(interval);
                }
            });

            while (!stoken.stop_requested())
            {
                std::this_thread::sleep_for(interval);
            }
        });

        std::this_thread::sleep_for(4 * interval);
        t.request_stop();
        PIKA_TEST(t.get_stop_source().stop_requested());
        t.join();
    }
}

////////////////////////////////////////////////////////////////////////////////
void test_basic_api_with_func()
{
    pika::stop_source ssource;
    PIKA_TEST(ssource.stop_possible());
    PIKA_TEST(!ssource.stop_requested());

    {
        pika::jthread t([]() {});
        ssource = t.get_stop_source();
        PIKA_TEST(ssource.stop_possible());
        PIKA_TEST(!ssource.stop_requested());
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    PIKA_TEST(ssource.stop_possible());
    PIKA_TEST(ssource.stop_requested());
}

////////////////////////////////////////////////////////////////////////////////
void test_exchange_token()
{
    auto interval = std::chrono::milliseconds(500);

    {
        std::atomic<pika::stop_token*> pstoken(nullptr);
        pika::jthread t([&pstoken](pika::stop_token sstoken) {
            auto act_token = sstoken;
            int num_interrupts = 0;
            try
            {
                for (int i = 0; num_interrupts < 2 && i < 500; ++i)
                {
                    // if we get a new interrupt token from the caller, take it
                    if (pstoken.load() != nullptr)
                    {
                        act_token = *pstoken;
                        if (act_token.stop_requested())
                        {
                            ++num_interrupts;
                        }
                        pstoken.store(nullptr);
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            catch (...)
            {
                PIKA_TEST(false);
            }
        });

        std::this_thread::sleep_for(interval);
        t.request_stop();

        std::this_thread::sleep_for(interval);
        pika::stop_token it;
        pstoken.store(&it);

        std::this_thread::sleep_for(interval);
        auto ssource2 = pika::stop_source{};
        it = pika::stop_token{ssource2.get_token()};
        pstoken.store(&it);

        std::this_thread::sleep_for(interval);
        ssource2.request_stop();

        std::this_thread::sleep_for(interval);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_concurrent_interrupt()
{
    int num_threads = 30;
    pika::stop_source is;

    {
        pika::jthread t1([it = is.get_token()](pika::stop_token stoken) {
            try
            {
                bool stop_requested = false;
                for (int i = 0; !it.stop_requested(); ++i)
                {
                    // should never switch back once requested
                    if (stoken.stop_requested())
                    {
                        stop_requested = true;
                    }
                    else
                    {
                        PIKA_TEST(!stop_requested);
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                PIKA_TEST(stop_requested);
            }
            catch (...)
            {
                PIKA_TEST(false);
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // starts thread concurrently calling request_stop() for the same token
        std::vector<pika::jthread> tv;
        int num_requested_stops = 0;
        for (int i = 0; i < num_threads; ++i)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            pika::jthread t([&t1, &num_requested_stops] {
                for (int i = 0; i < 13; ++i)
                {
                    // only first call to request_stop should return true
                    num_requested_stops += (t1.request_stop() ? 1 : 0);
                    PIKA_TEST(!t1.request_stop());
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
            tv.push_back(std::move(t));
        }

        for (auto& t : tv)
        {
            t.join();
        }

        // only one request to request_stop() should have returned true
        PIKA_TEST_EQ(num_requested_stops, 1);
        is.request_stop();
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_jthread_move()
{
    {
        bool interrupt_signalled = false;
        pika::jthread t{[&interrupt_signalled](pika::stop_token st) {
            while (!st.stop_requested())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (st.stop_requested())
            {
                interrupt_signalled = true;
            }
        }};

        pika::jthread t2{std::move(t)};    // should compile

        // NOLINTNEXTLINE(bugprone-use-after-move)
        auto ssource = t.get_stop_source();
        PIKA_TEST(!ssource.stop_possible());
        PIKA_TEST(!ssource.stop_requested());

        ssource = t2.get_stop_source();
        PIKA_TEST(ssource != pika::stop_source{});
        PIKA_TEST(ssource.stop_possible());
        PIKA_TEST(!ssource.stop_requested());

        PIKA_TEST(!interrupt_signalled);
        t.request_stop();
        PIKA_TEST(!interrupt_signalled);
        t2.request_stop();
        t2.join();
        PIKA_TEST(interrupt_signalled);
    }
}

///////////////////////////////////////////////////////////////////////////////
// void testEnabledIfForCopyConstructor_CompileTimeOnly()
// {
//     {
//         pika::jthread t;
//         //pika::jthread t2{t};  // should not compile
//     }
// }

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::set_terminate([]() { PIKA_TEST(false); });

    test_interrupt_by_destructor();
    test_interrupt_started_thread();
    test_interrupt_started_thread_with_subthread();
    test_basic_api_with_func();
    test_exchange_token();
    test_concurrent_interrupt();
    test_jthread_move();
    //     testEnabledIfForCopyConstructor_CompileTimeOnly();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
