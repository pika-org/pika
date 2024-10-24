//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <fmt/printf.h>

#include <string>
#include <tuple>
#include <utility>

int main(int argc, char* argv[])
{
    namespace ex = pika::execution::experimental;
    namespace tt = pika::this_thread::experimental;

    pika::start(argc, argv);
    ex::thread_pool_scheduler sched{};

    auto tuple_sender = ex::just(std::tuple(std::string("hello!"), 42)) | ex::continues_on(sched);
    auto process_data = [](auto message, auto answer) {
        fmt::print("{}\nthe answer is: {}\n", message, answer);
    };

    // With the unpack adaptor, process_data does not have to know that the data was originally sent
    // as a tuple
    auto unpack_sender = tuple_sender | ex::unpack() | ex::then(process_data);

    // We can manually recreate the behaviour of the unpack adaptor by using std::apply. This is
    // equivalent to the above.
    auto apply_sender = tuple_sender | ex::then([&](auto tuple_of_data) {
        return std::apply(process_data, std::move(tuple_of_data));
    });

    tt::sync_wait(ex::when_all(std::move(unpack_sender), std::move(apply_sender)));

    pika::finalize();
    pika::stop();

    return 0;
}
