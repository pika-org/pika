////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/assert.hpp>
#include <pika/execution.hpp>
#include <pika/functional/bind_front.hpp>
#include <pika/init.hpp>
#include <pika/latch.hpp>
#include <pika/modules/synchronization.hpp>

#include <cstddef>
#include <functional>
#include <iostream>
#include <sstream>

namespace ex = pika::execution::experimental;

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
void worker(std::size_t i, data& d, pika::latch& l)
{
    d.init.wait();
    std::ostringstream s;
    s << d.msg << ": " << i << "\n";
    std::cout << s.str();
    l.count_down(1);
}

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    data d;
    constexpr std::size_t num_tasks = 10;
    pika::latch l(num_tasks + 1);

    for (std::size_t i = 0; i < num_tasks; ++i)
        ex::execute(ex::thread_pool_scheduler{},
            pika::util::detail::bind_front(worker, i, std::ref(d), std::ref(l)));

    d.initialize("initialized");

    l.arrive_and_wait();

    return pika::finalize();
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
