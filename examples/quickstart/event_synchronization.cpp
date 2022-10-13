////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/assert.hpp>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/modules/synchronization.hpp>

#include <cstddef>
#include <functional>
#include <iostream>

struct data
{
    ///< For synchronizing two-phase initialization.
    pika::experimental::event init;

    char const* msg;

    data()
      : init()
      , msg("uninitialized")
    {
    }

    void initialize(char const* p)
    {
        // We can only be called once.
        PIKA_ASSERT(!init.occurred());
        msg = p;
        init.set();
    }
};

///////////////////////////////////////////////////////////////////////////////
void worker(std::size_t i, data& d, pika::counting_semaphore<>& sem)
{
    d.init.wait();
    std::cout << d.msg << ": " << i << "\n" << std::flush;
    sem.release();    // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    data d;
    pika::counting_semaphore<> sem{0};

    for (std::size_t i = 0; i < 10; ++i)
        pika::apply(&worker, i, std::ref(d), std::ref(sem));

    d.initialize("initialized");    // signal the event

    for (std::size_t i = 0; i < 10; ++i)
        sem.acquire();

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    return pika::init(pika_main, argc, argv);
}
