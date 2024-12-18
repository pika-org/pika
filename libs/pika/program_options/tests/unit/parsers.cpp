//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/detail/filesystem.hpp>
#include <pika/program_options/option.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/parsers.hpp>
#include <pika/program_options/positional_options.hpp>
#include <pika/program_options/value_semantic.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <cstdlib>    // for putenv
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace pika::program_options;
using namespace std;

pair<string, vector<vector<string>>> msp(string const& s1)
{
    return std::make_pair(s1, vector<vector<string>>());
}

pair<string, vector<vector<string>>> msp(string const& s1, string const& s2)
{
    vector<vector<string>> v(1);
    v[0].push_back(s2);
    return std::make_pair(s1, v);
}

void check_value(option const& option, char const* name, char const* value)
{
    PIKA_TEST_EQ(option.string_key, name);
    PIKA_TEST_EQ(option.value.size(), std::size_t(1));
    PIKA_TEST_EQ(option.value.front(), value);
}

vector<string> sv(char const* array[], unsigned size)
{
    vector<string> r;
    for (unsigned i = 0; i < size; ++i) r.emplace_back(array[i]);
    return r;
}

pair<string, string> additional_parser(std::string const&) { return pair<string, string>(); }

namespace command_line {

    void test_many_different_options()
    {
        // the long_names() API function was introduced in Boost V1.68
        options_description desc;
        desc.add_options()("foo,f", new untyped_value(), "")("bar,b", value<std::string>(), "")(
            "car,voiture", new untyped_value())("dog,dawg", new untyped_value())(
            "baz", new untyped_value())("plug*", new untyped_value());
        char const* cmdline3_[] = {"--foo=12", "-f4", "--bar=11", "-b4", "--voiture=15",
            "--dawg=16", "--dog=17", "--plug3=10"};
        vector<string> cmdline3 = sv(cmdline3_, sizeof(cmdline3_) / sizeof(char const*));
        vector<option> a3 = command_line_parser(cmdline3).options(desc).run().options;
        PIKA_TEST_EQ(a3.size(), 8u);
        check_value(a3[0], "foo", "12");
        check_value(a3[1], "foo", "4");
        check_value(a3[2], "bar", "11");
        check_value(a3[3], "bar", "4");
        check_value(a3[4], "car", "15");
        check_value(a3[5], "dog", "16");
        check_value(a3[6], "dog", "17");
        check_value(a3[7], "plug3", "10");

        // Regression test: check that '0' as style is interpreted as
        // 'default_style'
        vector<option> a4 = parse_command_line(
            sizeof(cmdline3_) / sizeof(char const*), cmdline3_, desc, 0, additional_parser)
                                .options;
        // The default style is unix-style, where the first argument on the command-line
        // is the name of a binary, not an option value, so that should be ignored
        PIKA_TEST_EQ(a4.size(), 7u);
        check_value(a4[0], "foo", "4");
        check_value(a4[1], "bar", "11");
        check_value(a4[2], "bar", "4");
        check_value(a4[3], "car", "15");
        check_value(a4[4], "dog", "16");
        check_value(a4[5], "dog", "17");
        check_value(a4[6], "plug3", "10");
    }

    void test_not_crashing_with_empty_string_values()
    {
        // Check that we don't crash on empty values of type 'string'
        char const* cmdline4[] = {"", "--open", ""};
        options_description desc2;
        desc2.add_options()("open", value<string>());
        variables_map vm;
        store(parse_command_line(
                  sizeof(cmdline4) / sizeof(char const*), const_cast<char**>(cmdline4), desc2),
            vm);
    }

    void test_multitoken()
    {
        char const* cmdline5[] = {"", "-p7", "-o", "1", "2", "3", "-x8"};
        options_description desc3;
        desc3.add_options()(",p", value<string>())(",o", value<string>()->multitoken())(
            ",x", value<string>());
        vector<option> a5 = parse_command_line(sizeof(cmdline5) / sizeof(char const*),
            const_cast<char**>(cmdline5), desc3, 0, additional_parser)
                                .options;
        PIKA_TEST_EQ(a5.size(), 3u);
        check_value(a5[0], "-p", "7");
        PIKA_TEST_EQ(a5[1].value.size(), std::size_t(3));
        PIKA_TEST_EQ(a5[1].string_key, "-o");
        PIKA_TEST_EQ(a5[1].value[0], "1");
        PIKA_TEST_EQ(a5[1].value[1], "2");
        PIKA_TEST_EQ(a5[1].value[2], "3");
        check_value(a5[2], "-x", "8");
    }

    void test_multitoken_and_multiname()
    {
        // the long_names() API function was introduced in Boost V1.68
        char const* cmdline[] = {
            "program", "-fone", "-b", "two", "--foo", "three", "four", "-zfive", "--fee", "six"};
        options_description desc;
        desc.add_options()("bar,b", value<string>())("foo,fee,f", value<string>()->multitoken())(
            "fizbaz,baz,z", value<string>());

        vector<option> parsed_options = parse_command_line(sizeof(cmdline) / sizeof(char const*),
            const_cast<char**>(cmdline), desc, 0, additional_parser)
                                            .options;

        PIKA_TEST_EQ(parsed_options.size(), 5u);
        check_value(parsed_options[0], "foo", "one");
        check_value(parsed_options[1], "bar", "two");
        PIKA_TEST_EQ(parsed_options[2].string_key, "foo");
        PIKA_TEST_EQ(parsed_options[2].value.size(), std::size_t(2));
        PIKA_TEST_EQ(parsed_options[2].value[0], "three");
        PIKA_TEST_EQ(parsed_options[2].value[1], "four");
        check_value(parsed_options[3], "fizbaz", "five");
        check_value(parsed_options[4], "foo", "six");

        char const* cmdline_2[] = {
            "program", "-fone", "-b", "two", "--fee", "three", "four", "-zfive", "--foo", "six"};

        parsed_options = parse_command_line(sizeof(cmdline_2) / sizeof(char const*),
            const_cast<char**>(cmdline_2), desc, 0, additional_parser)
                             .options;

        PIKA_TEST_EQ(parsed_options.size(), 5u);
        check_value(parsed_options[0], "foo", "one");
        check_value(parsed_options[1], "bar", "two");
        PIKA_TEST_EQ(parsed_options[2].string_key, "foo");
        PIKA_TEST_EQ(parsed_options[2].value.size(), std::size_t(2));
        PIKA_TEST_EQ(parsed_options[2].value[0], "three");
        PIKA_TEST_EQ(parsed_options[2].value[1], "four");
        check_value(parsed_options[3], "fizbaz", "five");
        check_value(parsed_options[4], "foo", "six");
    }

    void test_multitoken_vector_option()
    {
        // the long_names() API function was introduced in Boost V1.68
        options_description desc4("");
        desc4.add_options()("multitoken,multi-token,m",
            value<std::vector<std::string>>()->multitoken(),
            "values")("file", value<std::string>(), "the file to process");
        positional_options_description p;
        p.add("file", 1);
        char const* cmdline6[] = {"", "-m", "token1", "token2", "--", "some_file"};
        vector<option> a6 = command_line_parser(
            sizeof(cmdline6) / sizeof(char const*), const_cast<char**>(cmdline6))
                                .options(desc4)
                                .positional(p)
                                .run()
                                .options;
        PIKA_TEST_EQ(a6.size(), 2u);
        PIKA_TEST_EQ(a6[0].value.size(), std::size_t(2));
        PIKA_TEST_EQ(a6[0].string_key, "multitoken");
        PIKA_TEST_EQ(a6[0].value[0], "token1");
        PIKA_TEST_EQ(a6[0].value[1], "token2");
        PIKA_TEST_EQ(a6[1].string_key, "file");
        PIKA_TEST_EQ(a6[1].value.size(), std::size_t(1));
        PIKA_TEST_EQ(a6[1].value[0], "some_file");
    }

}    // namespace command_line

void test_command_line()
{
    command_line::test_many_different_options();
    // Check that we don't crash on empty values of type 'string'
    command_line::test_not_crashing_with_empty_string_values();
    command_line::test_multitoken();
    command_line::test_multitoken_vector_option();
    command_line::test_multitoken_and_multiname();
}

void test_config_file(char const* config_file)
{
    // the long_names() API function was introduced in Boost V1.68
    options_description desc;
    desc.add_options()("gv1", new untyped_value)("gv2", new untyped_value)(
        "empty_value", new untyped_value)("plug*", new untyped_value)("m1.v1", new untyped_value)(
        "m1.v2", new untyped_value)("m1.v3,alias3", new untyped_value)("b", bool_switch());

    char const content1[] = " gv1 = 0#asd\n"
                            "empty_value = \n"
                            "plug3 = 7\n"
                            "b = true\n"
                            "[m1]\n"
                            "v1 = 1\n"
                            "\n"
                            "v2 = 2\n"
                            "v3 = 3\n";

    stringstream ss(content1);
    vector<option> a1 = parse_config_file(ss, desc).options;
    PIKA_TEST_EQ(a1.size(), std::size_t(7));
    check_value(a1[0], "gv1", "0");
    check_value(a1[1], "empty_value", "");
    check_value(a1[2], "plug3", "7");
    check_value(a1[3], "b", "true");
    check_value(a1[4], "m1.v1", "1");
    check_value(a1[5], "m1.v2", "2");
    check_value(a1[6], "m1.v3", "3");

    // same test, but now options come from file
    vector<option> a2 = parse_config_file<char>(config_file, desc).options;
    PIKA_TEST_EQ(a2.size(), std::size_t(7));
    check_value(a2[0], "gv1", "0");
    check_value(a2[1], "empty_value", "");
    check_value(a2[2], "plug3", "7");
    check_value(a2[3], "b", "true");
    check_value(a2[4], "m1.v1", "1");
    check_value(a2[5], "m1.v2", "2");
    check_value(a2[6], "m1.v3", "3");
}

void test_environment()
{
    options_description desc;
    desc.add_options()("foo", new untyped_value, "")("bar", new untyped_value, "");

#if defined(_WIN32) || defined(__CYGWIN__)
    _putenv("PO_TEST_FOO=1");
#else
    putenv(const_cast<char*>("PO_TEST_FOO=1"));
#endif
    parsed_options p = parse_environment(desc, "PO_TEST_");

    PIKA_TEST_EQ(p.options.size(), std::size_t(1));
    PIKA_TEST_EQ(p.options[0].string_key, "foo");
    PIKA_TEST_EQ(p.options[0].value.size(), std::size_t(1));
    PIKA_TEST_EQ(p.options[0].value[0], "1");

    //TODO: since 'bar' does not allow a value, it cannot appear in environment,
    // which already has a value.
}

void test_unregistered()
{
    options_description desc;

    char const* cmdline1_[] = {"--foo=12", "--bar", "1"};
    vector<string> cmdline1 = sv(cmdline1_, sizeof(cmdline1_) / sizeof(char const*));
    vector<option> a1 =
        command_line_parser(cmdline1).options(desc).allow_unregistered().run().options;

    PIKA_TEST_EQ(a1.size(), std::size_t(3));
    PIKA_TEST_EQ(a1[0].string_key, "foo");
    PIKA_TEST_EQ(a1[0].unregistered, true);
    PIKA_TEST_EQ(a1[0].value.size(), std::size_t(1));
    PIKA_TEST_EQ(a1[0].value[0], "12");
    PIKA_TEST_EQ(a1[1].string_key, "bar");
    PIKA_TEST_EQ(a1[1].unregistered, true);
    PIKA_TEST_EQ(a1[2].string_key, "");
    PIKA_TEST_EQ(a1[2].unregistered, false);

    vector<string> a2 = collect_unrecognized(a1, include_positional);
    PIKA_TEST_EQ(a2[0], "--foo=12");
    PIKA_TEST_EQ(a2[1], "--bar");
    PIKA_TEST_EQ(a2[2], "1");

    // Test that storing unregistered options has no effect
    variables_map vm;

    store(command_line_parser(cmdline1).options(desc).allow_unregistered().run(), vm);

    PIKA_TEST_EQ(vm.size(), 0u);

    char const content1[] = "gv1 = 0\n"
                            "[m1]\n"
                            "v1 = 1\n";

    stringstream ss(content1);
    vector<option> a3 = parse_config_file(ss, desc, true).options;
    PIKA_TEST_EQ(a3.size(), std::size_t(2));
    check_value(a3[0], "gv1", "0");
    check_value(a3[1], "m1.v1", "1");
}

constexpr char const* const config_file = "config_test.cfg";
constexpr char const config_file_content[] = R"(
gv1 = 0#asd
empty_value =
plug3 = 7
b = true

[m1]
v1 = 1
v2 = 2
v3 = 3
)";

int main()
{
    test_command_line();
    test_environment();
    test_unregistered();

    // write config data to file
    {
        std::ofstream f(config_file, std::ios::out);
        f.write(config_file_content,
            (sizeof(config_file_content) / sizeof(config_file_content[0])) - 1);
    }

    test_config_file(config_file);

    // delete the config file
    std::filesystem::remove(config_file);

    return 0;
}
