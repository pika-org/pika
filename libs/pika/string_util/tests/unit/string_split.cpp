//  Copyright (c) 2020      ETH Zurich
//  Copyright (c) 2002-2003 Pavol Droba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/testing.hpp>
#include <pika/string_util/classification.hpp>
#include <pika/string_util/split.hpp>

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
    const char* pch1 = "xx-abc--xx-abb";
    std::vector<std::string> tokens;

    // split tests
    pika::string_util::split(tokens, str2, pika::string_util::is_any_of("xX"),
        pika::string_util::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(4));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[2], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[3], std::string(""));

    pika::string_util::split(tokens, str2, pika::string_util::is_any_of("xX"),
        pika::string_util::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[3], std::string(""));
    PIKA_TEST_EQ(tokens[4], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[5], std::string(""));
    PIKA_TEST_EQ(tokens[6], std::string(""));

    pika::string_util::split(tokens, pch1, pika::string_util::is_any_of("x"),
        pika::string_util::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(3));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[2], std::string("-abb"));

    pika::string_util::split(tokens, pch1, pika::string_util::is_any_of("x"),
        pika::string_util::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(5));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[3], std::string(""));
    PIKA_TEST_EQ(tokens[4], std::string("-abb"));

    pika::string_util::split(tokens, str3, pika::string_util::is_any_of(","),
        pika::string_util::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(1));
    PIKA_TEST_EQ(tokens[0], std::string("xx"));

    pika::string_util::split(tokens, str3, pika::string_util::is_any_of(","),
        pika::string_util::token_compress_mode::off);

    pika::string_util::split(tokens, str3, pika::string_util::is_any_of("xX"),
        pika::string_util::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(2));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));

    pika::string_util::split(tokens, str3, pika::string_util::is_any_of("xX"),
        pika::string_util::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(3));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string(""));

    split(tokens, strempty, pika::string_util::is_any_of(".:,;"),
        pika::string_util::token_compress_mode::on);

    PIKA_TEST(tokens.size() == 1);
    PIKA_TEST(tokens[0] == std::string(""));

    split(tokens, strempty, pika::string_util::is_any_of(".:,;"),
        pika::string_util::token_compress_mode::off);

    PIKA_TEST(tokens.size() == 1);
    PIKA_TEST(tokens[0] == std::string(""));

    // If using a compiler that supports forwarding references, we should be
    // able to use rvalues, too
    pika::string_util::split(tokens, std::string("Xx-abc--xX-abb-xx"),
        pika::string_util::is_any_of("xX"),
        pika::string_util::token_compress_mode::on);

    PIKA_TEST_EQ(tokens.size(), std::size_t(4));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[2], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[3], std::string(""));

    pika::string_util::split(tokens, std::string("Xx-abc--xX-abb-xx"),
        pika::string_util::is_any_of("xX"),
        pika::string_util::token_compress_mode::off);

    PIKA_TEST_EQ(tokens.size(), std::size_t(7));
    PIKA_TEST_EQ(tokens[0], std::string(""));
    PIKA_TEST_EQ(tokens[1], std::string(""));
    PIKA_TEST_EQ(tokens[2], std::string("-abc--"));
    PIKA_TEST_EQ(tokens[3], std::string(""));
    PIKA_TEST_EQ(tokens[4], std::string("-abb-"));
    PIKA_TEST_EQ(tokens[5], std::string(""));
    PIKA_TEST_EQ(tokens[6], std::string(""));

    return pika::util::report_errors();
}
