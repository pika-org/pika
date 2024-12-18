//  Copyright (c) 2020      ETH Zurich
//  Copyright (c) 2002-2003 Pavol Droba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/string_util/classification.hpp>
#include <pika/string_util/split.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <iostream>
#include <list>
#include <string>
#include <vector>

int main()
{
    std::string str2("Xx-abc--xX-abb-xx");
    std::string str3("xx");
    std::string strempty("");
    char const* pch1 = "xx-abc--xx-abb";
    std::vector<std::string> tokens;

    // split tests
    pika::detail::split(
        tokens, str2, pika::detail::is_any_of("xX"), pika::detail::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(4));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[2], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[3], std::string(""));

    pika::detail::split(
        tokens, str2, pika::detail::is_any_of("xX"), pika::detail::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[3], std::string(""));
    PIKA_TEST_EQ(tokens[4], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[5], std::string(""));
    PIKA_TEST_EQ(tokens[6], std::string(""));

    pika::detail::split(
        tokens, pch1, pika::detail::is_any_of("x"), pika::detail::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(3));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[2], std::string("-abb"));

    pika::detail::split(
        tokens, pch1, pika::detail::is_any_of("x"), pika::detail::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(5));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[3], std::string(""));
    PIKA_TEST_EQ(tokens[4], std::string("-abb"));

    pika::detail::split(
        tokens, str3, pika::detail::is_any_of(","), pika::detail::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(1));
    PIKA_TEST_EQ(tokens[0], std::string("xx"));

    pika::detail::split(
        tokens, str3, pika::detail::is_any_of(","), pika::detail::token_compress_mode::off);

    pika::detail::split(
        tokens, str3, pika::detail::is_any_of("xX"), pika::detail::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(2));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));

    pika::detail::split(
        tokens, str3, pika::detail::is_any_of("xX"), pika::detail::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(3));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string(""));

    split(tokens, strempty, pika::detail::is_any_of(".:,;"), pika::detail::token_compress_mode::on);

    PIKA_TEST(tokens.size() == 1);
    PIKA_TEST(tokens[0] == std::string(""));

    split(
        tokens, strempty, pika::detail::is_any_of(".:,;"), pika::detail::token_compress_mode::off);

    PIKA_TEST(tokens.size() == 1);
    PIKA_TEST(tokens[0] == std::string(""));

    // If using a compiler that supports forwarding references, we should be
    // able to use rvalues, too
    pika::detail::split(tokens, std::string("Xx-abc--xX-abb-xx"), pika::detail::is_any_of("xX"),
        pika::detail::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(4));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[2], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[3], std::string(""));

    pika::detail::split(tokens, std::string("Xx-abc--xX-abb-xx"), pika::detail::is_any_of("xX"),
        pika::detail::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[3], std::string(""));
    PIKA_TEST_EQ(tokens[4], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[5], std::string(""));
    PIKA_TEST_EQ(tokens[6], std::string(""));

    return 0;
}
