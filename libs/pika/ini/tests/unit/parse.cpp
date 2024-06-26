//  Copyright (c)   2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/testing.hpp>

#include <string>
#include <vector>

int main()
{
    std::vector<std::string> config = {
        "[system]", "pid=42", "[pika.stacks]", "small_stack_size=64"};
    pika::detail::section sec;
    sec.parse("<static defaults>", config, false, false, false);

    PIKA_TEST(sec.has_section("system"));
    PIKA_TEST(sec.has_section("pika.stacks"));
    PIKA_TEST(!sec.has_section("pika.thread_queue"));
}
