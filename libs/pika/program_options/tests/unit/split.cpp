//  Copyright Sascha Ochsenknecht 2009.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/program_options/cmdline.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/parsers.hpp>
#include <pika/program_options/value_semantic.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

using namespace pika::program_options;
using namespace std;

void check_value(string const& option, string const& value) { PIKA_TEST_EQ(option, value); }

void split_whitespace(options_description const& description)
{
    char const* cmdline = "prg --input input.txt \r --optimization 4  \t  --opt \n  option";

    vector<string> tokens = split_unix(cmdline, " \t\n\r");

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));

    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_equalsign(options_description const& description)
{
    char const* cmdline = "prg --input=input.txt  --optimization=4 --opt=option";

    vector<string> tokens = split_unix(cmdline, "= ");

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_semi(options_description const& description)
{
    char const* cmdline = "prg;--input input.txt;--optimization 4;--opt option";

    vector<string> tokens = split_unix(cmdline, "; ");

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_quotes(options_description const& description)
{
    char const* cmdline =
        R"(prg --input "input.txt input.txt" --optimization 4 --opt "option1 option2")";

    vector<string> tokens = split_unix(cmdline, " ");

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option1 option2");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_escape(options_description const& description)
{
    char const* cmdline =
        R"(prg --input \"input.txt\" --optimization 4 --opt \"option1\ option2\")";

    vector<string> tokens = split_unix(cmdline, " ");

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "\"input.txt\"");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "\"option1 option2\"");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_single_quote(options_description const& description)
{
    char const* cmdline =
        "prg --input 'input.txt input.txt' --optimization 4 --opt 'option1 option2'";

    vector<string> tokens = split_unix(cmdline, " ", "'");

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input.txt input.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "option1 option2");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

void split_defaults(options_description const& description)
{
    char const* cmdline =
        "prg --input \t \'input file.txt\' \t   --optimization 4 --opt \\\"option1\\ option2\\\"";

    vector<string> tokens = split_unix(cmdline);

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    check_value(tokens[0], "prg");
    check_value(tokens[1], "--input");
    check_value(tokens[2], "input file.txt");
    check_value(tokens[3], "--optimization");
    check_value(tokens[4], "4");
    check_value(tokens[5], "--opt");
    check_value(tokens[6], "\"option1 option2\"");

    variables_map vm;
    store(command_line_parser(tokens).options(description).run(), vm);
    notify(vm);
}

int main(int /*ac*/, char** /*av*/)
{
    options_description desc;
    // clang-format off
    desc.add_options()
        ("input,i", value<string>(),"the input file")
        ("optimization,O", value<unsigned>(), "optimization level")
        ("opt,o", value<string>(), "misc option")
        ;
    // clang-format on

    split_whitespace(desc);
    split_equalsign(desc);
    split_semi(desc);
    split_quotes(desc);
    split_escape(desc);
    split_single_quote(desc);
    split_defaults(desc);

    return 0;
}
