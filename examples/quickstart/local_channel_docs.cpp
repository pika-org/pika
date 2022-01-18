//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example is meant for inclusion in the documentation.

#include <pika/assert.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/synchronization.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void minimal_channel()
{
    //[local_channel_minimal
    pika::lcos::local::channel<int> c;
    pika::future<int> f = c.get();
    PIKA_ASSERT(!f.is_ready());
    c.set(42);
    PIKA_ASSERT(f.is_ready());
    std::cout << f.get() << std::endl;
    //]
}

///////////////////////////////////////////////////////////////////////////////
//[local_channel_send_receive
void do_something(pika::lcos::local::receive_channel<int> c,
    pika::lcos::local::send_channel<> done)
{
    // prints 43
    std::cout << c.get(pika::launch::sync) << std::endl;
    // signal back
    done.set();
}

void send_receive_channel()
{
    pika::lcos::local::channel<int> c;
    pika::lcos::local::channel<> done;

    pika::apply(&do_something, c, done);

    // send some value
    c.set(43);
    // wait for thread to be done
    done.get().wait();
}
//]

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    minimal_channel();
    send_receive_channel();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    return pika::local::init(pika_main, argc, argv);
}
