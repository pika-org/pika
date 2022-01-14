// Copyright Sascha Ochsenknecht 2009.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/testing.hpp>
#include <pika/program_options/cmdline.hpp>
#include <pika/program_options/detail/cmdline.hpp>
#include <pika/program_options/option.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/parsers.hpp>

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace pika::program_options;
using namespace std;

// Test free function collect_unrecognized()
//
//  it collects the tokens of all not registered options. It can be used
//  to pass them to an own parser implementation

void test_unrecognize_cmdline()
{
    options_description desc;

    string content = "prg --input input.txt --optimization 4 --opt option";
    vector<string> tokens = split_unix(content);

    pika::program_options::detail::cmdline cmd(tokens);
    cmd.set_options_description(desc);
    cmd.allow_unregistered();

    vector<option> opts = cmd.run();
    vector<string> result = collect_unrecognized(opts, include_positional);

    PIKA_TEST_EQ(result.size(), std::size_t(7));
    PIKA_TEST_EQ(result[0], "prg");
    PIKA_TEST_EQ(result[1], "--input");
    PIKA_TEST_EQ(result[2], "input.txt");
    PIKA_TEST_EQ(result[3], "--optimization");
    PIKA_TEST_EQ(result[4], "4");
    PIKA_TEST_EQ(result[5], "--opt");
    PIKA_TEST_EQ(result[6], "option");
}

void test_unrecognize_config()
{
    options_description desc;

    string content = " input = input.txt\n"
                     " optimization = 4\n"
                     " opt = option\n";

    stringstream ss(content);
    vector<option> opts = parse_config_file(ss, desc, true).options;
    vector<string> result = collect_unrecognized(opts, include_positional);

    PIKA_TEST_EQ(result.size(), std::size_t(6));
    PIKA_TEST_EQ(result[0], "input");
    PIKA_TEST_EQ(result[1], "input.txt");
    PIKA_TEST_EQ(result[2], "optimization");
    PIKA_TEST_EQ(result[3], "4");
    PIKA_TEST_EQ(result[4], "opt");
    PIKA_TEST_EQ(result[5], "option");
}

int main(int /*ac*/, char** /*av*/)
{
    test_unrecognize_cmdline();
    test_unrecognize_config();

    return 0;
}
