//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <pika/functional/bind_back.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/synchronization.hpp>
#include <pika/modules/testing.hpp>
#include <pika/modules/threading.hpp>

#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// helper to call wait()
void cv_wait(pika::stop_token stoken, int /* id */, bool& ready,
    pika::lcos::local::mutex& ready_mtx,
    pika::lcos::local::condition_variable_any& ready_cv, bool notify_called)
{
    try
    {
        {
            std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
            ready_cv.wait(lg, stoken, [&ready] { return ready; });
            if (stoken.stop_requested())
            {
                throw "interrupted";
            }
        }
        PIKA_TEST(!stoken.stop_requested());
        PIKA_TEST(notify_called);
    }
    catch (const char* e)
    {
        PIKA_TEST(!notify_called);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_cv(bool call_notify)
{
    bool ready = false;
    pika::lcos::local::mutex ready_mtx;
    pika::lcos::local::condition_variable_any ready_cv;

    {
        pika::jthread t1(
            [&ready, &ready_mtx, &ready_cv, call_notify](pika::stop_token it) {
                {
                    std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                    while (!it.stop_requested() && !ready)
                    {
                        ready_cv.wait_for(lg, std::chrono::milliseconds(100));
                    }
                }
                PIKA_TEST(call_notify != it.stop_requested());
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (call_notify)
        {
            {
                std::lock_guard<pika::lcos::local::mutex> lg(ready_mtx);
                ready = true;
            }    // release lock

            ready_cv.notify_one();
            t1.join();
        }
    }    // leave scope of t1 without join() or detach() (signals cancellation)
}

///////////////////////////////////////////////////////////////////////////////
void test_cv_pred(bool call_notify)
{
    bool ready = false;
    pika::lcos::local::mutex ready_mtx;
    pika::lcos::local::condition_variable_any ready_cv;

    {
        pika::jthread t1(
            [&ready, &ready_mtx, &ready_cv, call_notify](pika::stop_token st) {
                try
                {
                    std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                    ready_cv.wait(lg, st, [&ready] { return ready; });
                    if (st.stop_requested())
                    {
                        throw "interrupted";
                    }
                    PIKA_TEST(call_notify);
                }
                catch (std::exception const&)
                {
                    // should be no std::exception
                    PIKA_TEST(false);
                }
                catch (const char*)
                {
                    PIKA_TEST(!call_notify);
                }
                catch (...)
                {
                    PIKA_TEST(false);
                }
                PIKA_TEST(call_notify != st.stop_requested());
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (call_notify)
        {
            {
                std::lock_guard<pika::lcos::local::mutex> lg(ready_mtx);
                ready = true;
            }    // release lock

            ready_cv.notify_one();
            t1.join();
        }
    }    // leave scope of t1 without join() or detach() (signals cancellation)
}

///////////////////////////////////////////////////////////////////////////////
void test_cv_thread_no_pred(bool call_notify)
{
    bool ready = false;
    pika::lcos::local::mutex ready_mtx;
    pika::lcos::local::condition_variable_any ready_cv;

    pika::stop_source is;
    {
        pika::thread t1([&ready, &ready_mtx, &ready_cv, st = is.get_token(),
                           call_notify] {
            {
                std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                bool ret = ready_cv.wait(lg, st, [&ready] { return ready; });
                if (ret)
                {
                    PIKA_TEST(!st.stop_requested());
                    PIKA_TEST(call_notify);
                }
                else if (st.stop_requested())
                {
                    PIKA_TEST(!call_notify);
                }
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        PIKA_TEST(!is.stop_requested());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (call_notify)
        {
            {
                std::lock_guard<pika::lcos::local::mutex> lg(ready_mtx);
                ready = true;
            }    // release lock

            ready_cv.notify_one();
        }
        else
        {
            is.request_stop();
        }
        t1.join();
    }    // leave scope of t1 without join() or detach() (signals cancellation)
}

///////////////////////////////////////////////////////////////////////////////
void test_cv_thread_pred(bool call_notify)
{
    bool ready = false;
    pika::lcos::local::mutex ready_mtx;
    pika::lcos::local::condition_variable_any ready_cv;

    pika::stop_source is;
    {
        pika::thread t1(
            [&ready, &ready_mtx, &ready_cv, st = is.get_token(), call_notify] {
                bool ret;
                {
                    std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                    ret = ready_cv.wait(lg, st, [&ready] { return ready; });
                    if (ret)
                    {
                        PIKA_TEST(!st.stop_requested());
                    }
                    else
                    {
                        PIKA_TEST(st.stop_requested());
                    }
                }
                PIKA_TEST(call_notify != st.stop_requested());
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        PIKA_TEST(!is.stop_requested());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (call_notify)
        {
            {
                std::lock_guard<pika::lcos::local::mutex> lg(ready_mtx);
                ready = true;
            }    // release lock

            ready_cv.notify_one();
        }
        else
        {
            is.request_stop();
        }
        t1.join();
    }    // leave scope of t1 without join() or detach() (signals cancellation)
}

///////////////////////////////////////////////////////////////////////////////
void test_minimal_wait(int sec)
{
    // duration until interrupt is called
    auto dur = std::chrono::seconds{sec};

    try
    {
        bool ready = false;
        pika::lcos::local::mutex ready_mtx;
        pika::lcos::local::condition_variable_any ready_cv;

        {
            pika::jthread t1([&ready, &ready_mtx, &ready_cv, dur](
                                pika::stop_token st) {
                try
                {
                    auto t0 = std::chrono::steady_clock::now();
                    {
                        std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                        ready_cv.wait(lg, st, [&ready] { return ready; });
                    }
                    PIKA_TEST(std::chrono::steady_clock::now() <
                        t0 + dur + std::chrono::seconds(1));
                }
                catch (...)
                {
                    PIKA_TEST(false);
                }
            });

            std::this_thread::sleep_for(dur);
        }    // leave scope of t1 without join() or detach() (signals cancellation)
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_minimal_wait_for(int sec1, int sec2)
{
    // test the basic timed CV wait API
    auto dur_int = std::chrono::seconds{sec1};
    auto dur_wait = std::chrono::seconds{sec2};

    try
    {
        bool ready = false;
        pika::lcos::local::mutex ready_mtx;
        pika::lcos::local::condition_variable_any ready_cv;

        {
            pika::jthread t1([&ready, &ready_mtx, &ready_cv, dur_int, dur_wait](
                                pika::stop_token st) {
                try
                {
                    auto t0 = std::chrono::steady_clock::now();
                    {
                        std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                        ready_cv.wait_for(
                            lg, st, dur_wait, [&ready] { return ready; });
                    }
                    PIKA_TEST(std::chrono::steady_clock::now() <
                        t0 + dur_int + std::chrono::seconds(5));
                    PIKA_TEST(std::chrono::steady_clock::now() <
                        t0 + dur_wait + std::chrono::seconds(5));
                }
                catch (...)
                {
                    PIKA_TEST(false);
                }
            });

            std::this_thread::sleep_for(dur_int);
        }    // leave scope of t1 without join() or detach() (signals cancellation)
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Dur>
void test_timed_cv(bool call_notify, bool /* call_interrupt */, Dur dur)
{
    // test the basic jthread API
    bool ready = false;
    pika::lcos::local::mutex ready_mtx;
    pika::lcos::local::condition_variable_any ready_cv;

    {
        pika::jthread t1([&ready, &ready_mtx, &ready_cv, call_notify, dur](
                            pika::stop_token st) {
            auto t0 = std::chrono::steady_clock::now();
            int times_done{0};
            while (times_done < 3)
            {
                {
                    std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                    auto ret = ready_cv.wait_for(
                        lg, st, dur, [&ready] { return ready; });
                    if (dur > std::chrono::seconds(5))
                    {
                        PIKA_TEST(std::chrono::steady_clock::now() < t0 + dur);
                    }
                    if (ret)
                    {
                        PIKA_TEST(ready);
                        PIKA_TEST(!st.stop_requested());
                        PIKA_TEST(call_notify);
                        ++times_done;
                    }
                    else if (st.stop_requested())
                    {
                        PIKA_TEST(!ready);
                        PIKA_TEST(!call_notify);
                        ++times_done;
                    }
                }
            }
        });

        PIKA_TEST(!t1.get_stop_source().stop_requested());

        if (call_notify)
        {
            {
                std::lock_guard<pika::lcos::local::mutex> lg(ready_mtx);
                ready = true;
            }    // release lock

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            pika::this_thread::yield();
            ready_cv.notify_one();
        }
        else
        {
            t1.request_stop();
        }
    }    // leave scope of t1 without join() or detach() (signals cancellation)
}

///////////////////////////////////////////////////////////////////////////////
template <typename Dur>
void test_timed_wait(bool call_notify, bool call_interrupt, Dur dur)
{
    // test the basic jthread API

    // ready_ instead of ready to not clash with state::ready and work around
    // potential bug in GCC 10
    bool ready_ = false;
    pika::lcos::local::mutex ready_mtx;
    pika::lcos::local::condition_variable_any ready_cv;

    enum class state
    {
        loop,
        ready,
        interrupted
    };

    state t1_feedback{state::loop};
    {
        pika::jthread t1([&ready_, &ready_mtx, &ready_cv, call_notify, dur,
                            &t1_feedback](pika::stop_token st) {
            auto t0 = std::chrono::steady_clock::now();
            int times_done{0};
            while (times_done < 3)
            {
                try
                {
                    std::unique_lock<pika::lcos::local::mutex> lg{ready_mtx};
                    auto ret = ready_cv.wait_for(
                        lg, st, dur, [&ready_] { return ready_; });
                    if (st.stop_requested())
                    {
                        throw "interrupted";
                    }
                    if (dur > std::chrono::seconds(5))
                    {
                        PIKA_TEST(std::chrono::steady_clock::now() < t0 + dur);
                    }

                    if (ret)
                    {
                        t1_feedback = state::ready;
                        PIKA_TEST(ready_);
                        PIKA_TEST(!st.stop_requested());
                        PIKA_TEST(call_notify);
                        ++times_done;
                    }
                }
                catch (const char*)
                {
                    t1_feedback = state::interrupted;
                    PIKA_TEST(!ready_);
                    PIKA_TEST(!call_notify);
                    ++times_done;
                    if (times_done >= 3)
                    {
                        return;
                    }
                }
                catch (...)
                {
                    PIKA_TEST(false);
                }
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        PIKA_TEST(!t1.get_stop_source().stop_requested());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (call_notify)
        {
            {
                std::lock_guard<pika::lcos::local::mutex> lg(ready_mtx);
                ready_ = true;
            }    // release lock

            auto t0 = std::chrono::steady_clock::now();
            ready_cv.notify_one();
            while (t1_feedback != state::ready)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            PIKA_TEST(std::chrono::steady_clock::now() <
                t0 + std::chrono::seconds(5));
        }
        else if (call_interrupt)
        {
            auto t0 = std::chrono::steady_clock::now();
            t1.request_stop();
            while (t1_feedback != state::interrupted)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            PIKA_TEST(std::chrono::steady_clock::now() <
                t0 + std::chrono::seconds(5));
        }
    }    // leave scope of t1 without join() or detach() (signals cancellation)

    auto t0 = std::chrono::steady_clock::now();
    while (t1_feedback == state::loop)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    PIKA_TEST(std::chrono::steady_clock::now() < t0 + std::chrono::seconds(5));
}

///////////////////////////////////////////////////////////////////////////////
template <int NumExtraCV>
void test_many_cvs(bool call_notify, bool call_interrupt)
{
    {
        // thread t0 with CV:
        bool ready = false;
        pika::lcos::local::mutex ready_mtx;
        pika::lcos::local::condition_variable_any ready_cv;

        // don't forget to initialize with {} here !!!
        std::array<bool, NumExtraCV> arr_ready{};
        std::array<pika::lcos::local::mutex, NumExtraCV> arr_ready_mtx{};
        std::array<pika::lcos::local::condition_variable_any, NumExtraCV>
            arr_ready_cv{};
        std::vector<pika::jthread> vthreads_deferred;

        pika::jthread t0(pika::util::bind_back(cv_wait, 0, std::ref(ready),
            std::ref(ready_mtx), std::ref(ready_cv), call_notify));
        {
            auto t0ssource = t0.get_stop_source();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // starts thread concurrently calling request_stop() for the same
            // token
            std::vector<pika::jthread> vthreads;
            for (int idx = 0; idx < NumExtraCV; ++idx)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                pika::jthread t([idx, t0stoken = t0ssource.get_token(),
                                   &arr_ready, &arr_ready_mtx, &arr_ready_cv,
                                   call_notify] {
                    // use interrupt token of t0 instead
                    // NOTE: disables signaling interrupts directly to the thread
                    cv_wait(t0stoken, idx + 1, arr_ready[idx],
                        arr_ready_mtx[idx], arr_ready_cv[idx], call_notify);
                });
                vthreads.push_back(std::move(t));
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (call_notify)
            {
                {
                    std::lock_guard<pika::lcos::local::mutex> lg(ready_mtx);
                    ready = true;
                }    // release lock

                ready_cv.notify_one();
                t0.join();

                for (int idx = 0; idx < NumExtraCV; ++idx)
                {
                    {
                        std::lock_guard<pika::lcos::local::mutex> lg(
                            arr_ready_mtx[idx]);
                        arr_ready[idx] = true;
                        arr_ready_cv[idx].notify_one();
                    }    // release lock
                }
            }
            else if (call_interrupt)
            {
                t0.request_stop();
            }
            else
            {
                // Move ownership of the threads to a scope that will
                // destruct after thread t0 has destructed (and thus
                // signaled cancellation and joined) but before the
                // condition_variable/mutex/ready-flag objects that
                // they reference have been destroyed.
                vthreads_deferred = std::move(vthreads);
            }
        }    // leave scope of additional threads (already notified/interrupted
             // or detached)
    }    // leave scope of t0 without join() or detach() (signals cancellation)
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    std::set_terminate([]() { PIKA_TEST(false); });

    try
    {
        test_minimal_wait(0);
        test_minimal_wait(1);
        test_minimal_wait_for(0, 0);
        test_minimal_wait_for(0, 2);    // 0s for interrupt, 2s for wait
        test_minimal_wait_for(2, 0);    // 2s for interrupt, 0s for wait
        test_minimal_wait_for(1, 3);    // 1s for interrupt, 3s for wait
        test_minimal_wait_for(3, 1);    // 3s for interrupt, 1s for wait

        test_cv_thread_no_pred(false);    // signal cancellation
        test_cv_thread_no_pred(true);     // call notify()

        test_cv(false);    // signal cancellation
        test_cv(true);     // call notify()

        test_cv_pred(false);    // signal cancellation
        test_cv_pred(true);     // call notify()

        test_cv_thread_pred(false);    // signal cancellation
        test_cv_thread_pred(true);     // call notify()

        // call notify(), don't call request_stop()
        test_timed_cv(true, false, std::chrono::milliseconds(200));
        // don't call notify, call request_stop()
        test_timed_cv(false, true, std::chrono::milliseconds(200));
        // don't call notify, don't call request_stop()
        test_timed_cv(false, false, std::chrono::milliseconds(200));
        //         // call notify(), don't call request_stop()
        //         test_timed_cv(true, false, 60s);
        //         // don't call notify, call request_stop()
        //         test_timed_cv(false, true, 60s);
        //         // don't call notify, don't call request_stop()
        //         test_timed_cv(false, false, 60s);

        // call notify(), don't call request_stop()
        test_timed_wait(true, false, std::chrono::milliseconds(200));
        // don't call notify, call request_stop()
        test_timed_wait(false, true, std::chrono::milliseconds(200));
        // don't call notify, don't call request_stop()
        test_timed_wait(false, false, std::chrono::milliseconds(200));
        // call notify(), don't call request_stop()
        //         test_timed_wait(true, false, 60s);
        //         // don't call notify, call request_stop()
        //         test_timed_wait(false, true, 60s);
        //         // don't call notify, don't call request_stop()
        //         test_timed_wait(false, false, 60s);

        // call notify(), don't call request_stop()
        test_many_cvs<9>(true, false);
        // don't call notify, call request_stop()
        test_many_cvs<9>(false, true);
        // don't call notify, don't call request_stop() (implicit interrupt)
        test_many_cvs<9>(false, false);
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
