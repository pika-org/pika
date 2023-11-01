//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/string_util.hpp>

#include <fmt/printf.h>

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

struct pipeline
{
    static void process(std::vector<std::string> const& input)
    {
        std::vector<ex::unique_any_sender<>> tasks;
        for (auto s : input)
        {
            auto sender = ex::transfer_just(ex::thread_pool_scheduler{}, "Error.*", std::move(s)) |
                ex::let_value([](std::string re, std::string item) -> ex::unique_any_sender<> {
                    std::regex regex(std::move(re));
                    if (std::regex_match(item, regex))
                    {
                        return ex::transfer_just(ex::thread_pool_scheduler{}, std::move(item)) |
                            ex::then([](std::string s) {
                                return pika::detail::trim_copy(std::move(s));
                            }) |
                            ex::then([](std::string_view tc) { fmt::print("->{}\n", tc); });
                    }
                    else { return ex::just(); }
                });

            tasks.push_back(std::move(sender));
        }

        tt::sync_wait(ex::when_all_vector(std::move(tasks)));
    }
};

int pika_main()
{
    std::string inputs[] = {
        "Error: foobar", "Error. foo", " Warning: barbaz", "Notice: qux", "\tError: abc"};
    std::vector<std::string> input(std::begin(inputs), std::end(inputs));

    pipeline::process(input);

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) { return pika::init(pika_main, argc, argv); }
