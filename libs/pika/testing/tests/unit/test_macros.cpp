//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/testing.hpp>

#include <sstream>
#include <string>

int main()
{
    std::stringstream strm;

    // testing macros
    PIKA_TEST(true);
    PIKA_TEST(strm, true);

    PIKA_TEST_MSG(true, "should be true");
    PIKA_TEST_MSG(strm, true, "should be true");

    PIKA_TEST_EQ(0, 0);
    PIKA_TEST_EQ(strm, 0, 0);

    PIKA_TEST_EQ_MSG(0, 0, "should be equal");
    PIKA_TEST_EQ_MSG(strm, 0, 0, "should be equal");

    PIKA_TEST_NEQ(0, 1);
    PIKA_TEST_NEQ(strm, 0, 1);

    PIKA_TEST_NEQ_MSG(0, 1, "should not be equal");
    PIKA_TEST_NEQ_MSG(strm, 0, 1, "should not be equal");

    PIKA_TEST_LT(0, 1);
    PIKA_TEST_LT(strm, 0, 1);

    PIKA_TEST_LT_MSG(0, 1, "should be less");
    PIKA_TEST_LT_MSG(strm, 0, 1, "should be less");

    PIKA_TEST_LTE(1, 1);
    PIKA_TEST_LTE(strm, 1, 1);

    PIKA_TEST_LTE_MSG(1, 1, "should be less equal");
    PIKA_TEST_LTE_MSG(strm, 1, 1, "should be less equal");

    PIKA_TEST_RANGE(1, 1, 1);
    PIKA_TEST_RANGE(strm, 1, 1, 1);

    PIKA_TEST_RANGE_MSG(1, 1, 1, "should be in range");
    PIKA_TEST_RANGE_MSG(strm, 1, 1, 1, "should be in range");

    // sanity macro tests
    PIKA_SANITY(true);
    PIKA_SANITY(strm, true);

    PIKA_SANITY_MSG(true, "should be true");
    PIKA_SANITY_MSG(strm, true, "should be true");

    PIKA_SANITY_EQ(0, 0);
    PIKA_SANITY_EQ(strm, 0, 0);

    PIKA_SANITY_EQ_MSG(0, 0, "should be equal");
    PIKA_SANITY_EQ_MSG(strm, 0, 0, "should be equal");

    PIKA_SANITY_NEQ(0, 1);
    PIKA_SANITY_NEQ(strm, 0, 1);

    PIKA_SANITY_LT(0, 1);
    PIKA_SANITY_LT(strm, 0, 1);

    PIKA_SANITY_LTE(1, 1);
    PIKA_SANITY_LTE(strm, 1, 1);

    PIKA_SANITY_RANGE(1, 1, 1);
    PIKA_SANITY_RANGE(strm, 1, 1, 1);

    // there shouldn't be any output being generated
    PIKA_TEST(strm.str().empty());

    // now test that something gets written to the stream if an error occurs
    PIKA_TEST(strm, false);
    PIKA_TEST(strm.str().find("test 'false'") != std::string::npos);

    // we have intentionally generated one error
    return (pika::util::report_errors() == 1) ? 0 : -1;
}
