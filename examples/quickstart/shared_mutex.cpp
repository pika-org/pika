//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example was released to the public domain by Stephan T. Lavavej
// (see: https://channel9.msdn.com/Shows/C9-GoingNative/GoingNative-40-Updated-STL-in-VS-2015-feat-STL)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/shared_mutex.hpp>
#include <pika/local/thread.hpp>
#include <pika/type_support/unused.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <vector>

int const writers = 3;
int const readers = 3;
int const cycles = 10;

using std::chrono::milliseconds;

int pika_main()
{
    std::vector<pika::thread> threads;
    std::atomic<bool> ready(false);
    pika::lcos::local::shared_mutex stm;

    for (int i = 0; i < writers; ++i)
    {
        threads.emplace_back([&ready, &stm, i] {
            std::mt19937 urng(static_cast<std::uint32_t>(std::time(nullptr)));
            std::uniform_int_distribution<int> dist(1, 1000);

            while (!ready)
            { /*** wait... ***/
            }

            for (int j = 0; j < cycles; ++j)
            {
                std::unique_lock<pika::lcos::local::shared_mutex> ul(stm);

                std::cout << "^^^ Writer " << i << " starting..." << std::endl;
                pika::this_thread::sleep_for(milliseconds(dist(urng)));
                std::cout << "vvv Writer " << i << " finished." << std::endl;

                ul.unlock();

                pika::this_thread::sleep_for(milliseconds(dist(urng)));
            }
        });
    }

    for (int i = 0; i < readers; ++i)
    {
        int k = writers + i;
        threads.emplace_back([&ready, &stm, k, i] {
            PIKA_UNUSED(k);
            std::mt19937 urng(static_cast<std::uint32_t>(std::time(nullptr)));
            std::uniform_int_distribution<int> dist(1, 1000);

            while (!ready)
            { /*** wait... ***/
            }

            for (int j = 0; j < cycles; ++j)
            {
                std::shared_lock<pika::lcos::local::shared_mutex> sl(stm);

                std::cout << "Reader " << i << " starting..." << std::endl;
                pika::this_thread::sleep_for(milliseconds(dist(urng)));
                std::cout << "Reader " << i << " finished." << std::endl;

                sl.unlock();

                pika::this_thread::sleep_for(milliseconds(dist(urng)));
            }
        });
    }

    ready = true;
    for (auto& t : threads)
        t.join();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
