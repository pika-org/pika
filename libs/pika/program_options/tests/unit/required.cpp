//  Copyright Sascha Ochsenknecht 2009.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/filesystem.hpp>
#include <pika/modules/program_options.hpp>
#include <pika/modules/testing.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace pika::program_options;
using namespace std;

void required_throw_test()
{
    options_description opts;
    // clang-format off
    opts.add_options()
        ("cfgfile,c", value<string>()->required(), "the configfile")
        ("fritz,f", value<string>()->required(), "the output file")
        ;
    // clang-format on

    variables_map vm;
    bool thrown = false;
    {
        // This test must throw exception
        string cmdline = "prg -f file.txt";
        vector<string> tokens = split_unix(cmdline);
        thrown = false;
        try
        {
            store(command_line_parser(tokens).options(opts).run(), vm);
            notify(vm);
        }
        catch (required_option& e)
        {
            PIKA_TEST_EQ(e.what(),
                string("the option '--cfgfile' is required but missing"));
            thrown = true;
        }
        PIKA_TEST(thrown);
    }

    {
        // This test mustn't throw exception
        string cmdline = "prg -c config.txt";
        vector<string> tokens = split_unix(cmdline);
        thrown = false;
        try
        {
            store(command_line_parser(tokens).options(opts).run(), vm);
            notify(vm);
        }
        catch (required_option const&)
        {
            thrown = true;
        }
        PIKA_TEST(!thrown);
    }
}

void simple_required_test(const char* config_file)
{
    options_description opts;
    // clang-format off
    opts.add_options()
        ("cfgfile,c", value<string>()->required(), "the configfile")
        ("fritz,f", value<string>()->required(), "the output file")
        ;
    // clang-format on

    variables_map vm;
    bool thrown = false;
    {
        // This test must throw exception
        string cmdline = "prg -f file.txt";
        vector<string> tokens = split_unix(cmdline);
        thrown = false;
        try
        {
            // options coming from different sources
            store(command_line_parser(tokens).options(opts).run(), vm);
            store(parse_config_file<char>(config_file, opts), vm);
            notify(vm);
        }
        catch (required_option const&)
        {
            thrown = true;
        }
        PIKA_TEST(!thrown);
    }
}

void multiname_required_test()
{
    // the long_names() API function was introduced in Boost V1.68
    options_description opts;
    opts.add_options()("foo,bar", value<string>()->required(), "the foo");

    variables_map vm;
    bool thrown = false;
    {
        // This test must throw exception
        string cmdline = "prg --bar file.txt";
        vector<string> tokens = split_unix(cmdline);
        thrown = false;
        try
        {
            // options coming from different sources
            store(command_line_parser(tokens).options(opts).run(), vm);
            notify(vm);
        }
        catch (required_option const&)
        {
            thrown = true;
        }
        PIKA_TEST(!thrown);
    }
}

constexpr char const* const config_file = "required_test.cfg";
constexpr char const config_file_content[] = R"(
cfgfile = file.cfg
)";

int main()
{
    required_throw_test();
    multiname_required_test();

    // write config data to file
    {
        std::ofstream f(config_file, std::ios::out);
        f.write(config_file_content,
            (sizeof(config_file_content) / sizeof(config_file_content[0])) - 1);
    }

    simple_required_test(config_file);

    // delete the config file
    pika::filesystem::remove(config_file);

    return pika::util::report_errors();
}
