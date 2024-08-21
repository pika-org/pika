////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/config.hpp>
#include <pika/preprocessor/cat.hpp>
#include <pika/preprocessor/expand.hpp>
#include <pika/preprocessor/nargs.hpp>
#include <pika/preprocessor/stringize.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/testing/detail.hpp>

#include <string>
#endif

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST(...)                                                                             \
    PIKA_TEST_(__VA_ARGS__)                                                                        \
    /**/

#define PIKA_TEST_(...)                                                                            \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))               \
    /**/
#define PIKA_TEST_1(expr) PIKA_TEST_IMPL(::pika::detail::global_fixture, expr)
#define PIKA_TEST_2(strm, expr) PIKA_TEST_IMPL(::pika::detail::fixture{strm}, expr)

#define PIKA_TEST_IMPL(fixture, expr)                                                              \
    fixture.check_(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION, ::pika::detail::counter_test, \
        expr, "test '" PIKA_PP_STRINGIZE(expr) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_MSG(...)                                                                         \
    PIKA_TEST_MSG_(__VA_ARGS__)                                                                    \
    /**/

#define PIKA_TEST_MSG_(...)                                                                        \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))           \
    /**/
#define PIKA_TEST_MSG_2(expr, msg) PIKA_TEST_MSG_IMPL(::pika::detail::global_fixture, expr, msg)
#define PIKA_TEST_MSG_3(strm, expr, msg)                                                           \
    PIKA_TEST_MSG_IMPL(::pika::detail::fixture{strm}, expr, msg)

#define PIKA_TEST_MSG_IMPL(fixture, expr, msg)                                                     \
    fixture.check_(                                                                                \
        __FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION, ::pika::detail::counter_test, expr, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_EQ(...)                                                                          \
    PIKA_TEST_EQ_(__VA_ARGS__)                                                                     \
    /**/

#define PIKA_TEST_EQ_(...)                                                                         \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_EQ_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))            \
    /**/
#define PIKA_TEST_EQ_2(expr1, expr2) PIKA_TEST_EQ_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_TEST_EQ_3(strm, expr1, expr2)                                                         \
    PIKA_TEST_EQ_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_TEST_EQ_IMPL(fixture, expr1, expr2)                                                   \
    fixture.check_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                          \
        ::pika::detail::counter_test, expr1, expr2,                                                \
        "test '" PIKA_PP_STRINGIZE(expr1) " == " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_NEQ(...)                                                                         \
    PIKA_TEST_NEQ_(__VA_ARGS__)                                                                    \
    /**/

#define PIKA_TEST_NEQ_(...)                                                                        \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_NEQ_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))           \
    /**/
#define PIKA_TEST_NEQ_2(expr1, expr2)                                                              \
    PIKA_TEST_NEQ_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_TEST_NEQ_3(strm, expr1, expr2)                                                        \
    PIKA_TEST_NEQ_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_TEST_NEQ_IMPL(fixture, expr1, expr2)                                                  \
    fixture.check_not_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                      \
        ::pika::detail::counter_test, expr1, expr2,                                                \
        "test '" PIKA_PP_STRINGIZE(expr1) " != " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_LT(...)                                                                          \
    PIKA_TEST_LT_(__VA_ARGS__)                                                                     \
    /**/

#define PIKA_TEST_LT_(...)                                                                         \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_LT_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))            \
    /**/
#define PIKA_TEST_LT_2(expr1, expr2) PIKA_TEST_LT_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_TEST_LT_3(strm, expr1, expr2)                                                         \
    PIKA_TEST_LT_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_TEST_LT_IMPL(fixture, expr1, expr2)                                                   \
    fixture.check_less(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                           \
        ::pika::detail::counter_test, expr1, expr2,                                                \
        "test '" PIKA_PP_STRINGIZE(expr1) " < " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_LTE(...)                                                                         \
    PIKA_TEST_LTE_(__VA_ARGS__)                                                                    \
    /**/

#define PIKA_TEST_LTE_(...)                                                                        \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_LTE_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))           \
    /**/
#define PIKA_TEST_LTE_2(expr1, expr2)                                                              \
    PIKA_TEST_LTE_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_TEST_LTE_3(strm, expr1, expr2)                                                        \
    PIKA_TEST_LTE_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_TEST_LTE_IMPL(fixture, expr1, expr2)                                                  \
    fixture.check_less_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                     \
        ::pika::detail::counter_test, expr1, expr2,                                                \
        "test '" PIKA_PP_STRINGIZE(expr1) " <= " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_RANGE(...)                                                                       \
    PIKA_TEST_RANGE_(__VA_ARGS__)                                                                  \
    /**/

#define PIKA_TEST_RANGE_(...)                                                                      \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_RANGE_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))         \
    /**/
#define PIKA_TEST_RANGE_3(expr1, expr2, expr3)                                                     \
    PIKA_TEST_RANGE_IMPL(::pika::detail::global_fixture, expr1, expr2, expr3)
#define PIKA_TEST_RANGE_4(strm, expr1, expr2, expr3)                                               \
    PIKA_TEST_RANGE_IMPL(::pika::detail::fixture{strm}, expr1, expr2, expr3)

#define PIKA_TEST_RANGE_IMPL(fixture, expr1, expr2, expr3)                                         \
    fixture.check_range(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                          \
        ::pika::detail::counter_test, expr1, expr2, expr3,                                         \
        "test '" PIKA_PP_STRINGIZE(expr2) " <= " PIKA_PP_STRINGIZE(                                \
            expr1) " <= " PIKA_PP_STRINGIZE(expr3) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_EQ_MSG(...)                                                                      \
    PIKA_TEST_EQ_MSG_(__VA_ARGS__)                                                                 \
    /**/

#define PIKA_TEST_EQ_MSG_(...)                                                                     \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_EQ_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))        \
    /**/
#define PIKA_TEST_EQ_MSG_3(expr1, expr2, msg)                                                      \
    PIKA_TEST_EQ_MSG_IMPL(::pika::detail::global_fixture, expr1, expr2, msg)
#define PIKA_TEST_EQ_MSG_4(strm, expr1, expr2, msg)                                                \
    PIKA_TEST_EQ_MSG_IMPL(::pika::detail::fixture{strm}, expr1, expr2, msg)

#define PIKA_TEST_EQ_MSG_IMPL(fixture, expr1, expr2, msg)                                          \
    fixture.check_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                          \
        ::pika::detail::counter_test, expr1, expr2, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_NEQ_MSG(...)                                                                     \
    PIKA_TEST_NEQ_MSG_(__VA_ARGS__)                                                                \
    /**/

#define PIKA_TEST_NEQ_MSG_(...)                                                                    \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_NEQ_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))       \
    /**/
#define PIKA_TEST_NEQ_MSG_3(expr1, expr2, msg)                                                     \
    PIKA_TEST_NEQ_MSG_IMPL(::pika::detail::global_fixture, expr1, expr2, msg)
#define PIKA_TEST_NEQ_MSG_4(strm, expr1, expr2, msg)                                               \
    PIKA_TEST_NEQ_MSG_IMPL(::pika::detail::fixture{strm}, expr1, expr2, msg)

#define PIKA_TEST_NEQ_MSG_IMPL(fixture, expr1, expr2, msg)                                         \
    fixture.check_not_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                      \
        ::pika::detail::counter_test, expr1, expr2, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_LT_MSG(...)                                                                      \
    PIKA_TEST_LT_MSG_(__VA_ARGS__)                                                                 \
    /**/

#define PIKA_TEST_LT_MSG_(...)                                                                     \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_LT_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))        \
    /**/
#define PIKA_TEST_LT_MSG_3(expr1, expr2, msg)                                                      \
    PIKA_TEST_LT_MSG_IMPL(::pika::detail::global_fixture, expr1, expr2, msg)
#define PIKA_TEST_LT_MSG_4(strm, expr1, expr2, msg)                                                \
    PIKA_TEST_LT_MSG_IMPL(::pika::detail::fixture{strm}, expr1, expr2, msg)

#define PIKA_TEST_LT_MSG_IMPL(fixture, expr1, expr2, msg)                                          \
    fixture.check_less(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                           \
        ::pika::detail::counter_test, expr1, expr2, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_LTE_MSG(...)                                                                     \
    PIKA_TEST_LTE_MSG_(__VA_ARGS__)                                                                \
    /**/

#define PIKA_TEST_LTE_MSG_(...)                                                                    \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_LTE_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))       \
    /**/
#define PIKA_TEST_LTE_MSG_3(expr1, expr2, msg)                                                     \
    PIKA_TEST_LTE_MSG_IMPL(::pika::detail::global_fixture, expr1, expr2, msg)
#define PIKA_TEST_LTE_MSG_4(strm, expr1, expr2, msg)                                               \
    PIKA_TEST_LTE_MSG_IMPL(::pika::detail::fixture{strm}, expr1, expr2, msg)

#define PIKA_TEST_LTE_MSG_IMPL(fixture, expr1, expr2, msg)                                         \
    fixture.check_less_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                     \
        ::pika::detail::counter_test, expr1, expr2, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_RANGE_MSG(...)                                                                   \
    PIKA_TEST_RANGE_MSG_(__VA_ARGS__)                                                              \
    /**/

#define PIKA_TEST_RANGE_MSG_(...)                                                                  \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_RANGE_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))     \
    /**/
#define PIKA_TEST_RANGE_MSG_4(expr1, expr2, expr3, msg)                                            \
    PIKA_TEST_RANGE_MSG_IMPL(::pika::detail::global_fixture, expr1, expr2, expr3, msg)
#define PIKA_TEST_RANGE_MSG_5(strm, expr1, expr2, expr3, msg)                                      \
    PIKA_TEST_RANGE_MSG_IMPL(::pika::detail::fixture{strm}, expr1, expr2, expr3, msg)

#define PIKA_TEST_RANGE_MSG_IMPL(fixture, expr1, expr2, expr3, msg)                                \
    fixture.check_range(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                          \
        ::pika::detail::counter_test, expr1, expr2, expr3, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY(...)                                                                           \
    PIKA_SANITY_(__VA_ARGS__)                                                                      \
    /**/

#define PIKA_SANITY_(...)                                                                          \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))             \
    /**/
#define PIKA_SANITY_1(expr) PIKA_TEST_IMPL(::pika::detail::global_fixture, expr)
#define PIKA_SANITY_2(strm, expr) PIKA_SANITY_IMPL(::pika::detail::fixture{strm}, expr)

#define PIKA_SANITY_IMPL(fixture, expr)                                                            \
    fixture.check_(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                               \
        ::pika::detail::counter_sanity, expr, "sanity check '" PIKA_PP_STRINGIZE(expr) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY_MSG(...)                                                                       \
    PIKA_SANITY_MSG_(__VA_ARGS__)                                                                  \
    /**/

#define PIKA_SANITY_MSG_(...)                                                                      \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))         \
    /**/
#define PIKA_SANITY_MSG_2(expr, msg) PIKA_SANITY_MSG_IMPL(::pika::detail::global_fixture, expr, msg)
#define PIKA_SANITY_MSG_3(strm, expr, msg)                                                         \
    PIKA_SANITY_MSG_IMPL(::pika::detail::fixture{strm}, expr, msg)

#define PIKA_SANITY_MSG_IMPL(fixture, expr, msg)                                                   \
    fixture.check_(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                               \
        ::pika::detail::counter_sanity, expr, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY_EQ(...)                                                                        \
    PIKA_SANITY_EQ_(__VA_ARGS__)                                                                   \
    /**/

#define PIKA_SANITY_EQ_(...)                                                                       \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_EQ_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))          \
    /**/
#define PIKA_SANITY_EQ_2(expr1, expr2)                                                             \
    PIKA_SANITY_EQ_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_SANITY_EQ_3(strm, expr1, expr2)                                                       \
    PIKA_SANITY_EQ_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_SANITY_EQ_IMPL(fixture, expr1, expr2)                                                 \
    fixture.check_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                          \
        ::pika::detail::counter_sanity, expr1, expr2,                                              \
        "sanity check '" PIKA_PP_STRINGIZE(expr1) " == " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY_NEQ(...)                                                                       \
    PIKA_SANITY_NEQ_(__VA_ARGS__)                                                                  \
    /**/

#define PIKA_SANITY_NEQ_(...)                                                                      \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_NEQ_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))         \
    /**/
#define PIKA_SANITY_NEQ_2(expr1, expr2)                                                            \
    PIKA_SANITY_NEQ_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_SANITY_NEQ_3(strm, expr1, expr2)                                                      \
    PIKA_SANITY_NEQ_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_SANITY_NEQ_IMPL(fixture, expr1, expr2)                                                \
    fixture.check_not_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                      \
        ::pika::detail::counter_sanity, expr1, expr2,                                              \
        "sanity check '" PIKA_PP_STRINGIZE(expr1) " != " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY_LT(...)                                                                        \
    PIKA_SANITY_LT_(__VA_ARGS__)                                                                   \
    /**/

#define PIKA_SANITY_LT_(...)                                                                       \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_LT_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))          \
    /**/
#define PIKA_SANITY_LT_2(expr1, expr2)                                                             \
    PIKA_SANITY_LT_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_SANITY_LT_3(strm, expr1, expr2)                                                       \
    PIKA_SANITY_LT_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_SANITY_LT_IMPL(fixture, expr1, expr2)                                                 \
    fixture.check_less(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                           \
        ::pika::detail::counter_sanity, expr1, expr2,                                              \
        "sanity check '" PIKA_PP_STRINGIZE(expr1) " < " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY_LTE(...)                                                                       \
    PIKA_SANITY_LTE_(__VA_ARGS__)                                                                  \
    /**/

#define PIKA_SANITY_LTE_(...)                                                                      \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_LTE_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))         \
    /**/
#define PIKA_SANITY_LTE_2(expr1, expr2)                                                            \
    PIKA_SANITY_LTE_IMPL(::pika::detail::global_fixture, expr1, expr2)
#define PIKA_SANITY_LTE_3(strm, expr1, expr2)                                                      \
    PIKA_SANITY_LTE_IMPL(::pika::detail::fixture{strm}, expr1, expr2)

#define PIKA_SANITY_LTE_IMPL(fixture, expr1, expr2)                                                \
    fixture.check_less_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                     \
        ::pika::detail::counter_sanity, expr1, expr2,                                              \
        "sanity check '" PIKA_PP_STRINGIZE(expr1) " <= " PIKA_PP_STRINGIZE(expr2) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY_RANGE(...)                                                                     \
    PIKA_SANITY_RANGE_(__VA_ARGS__)                                                                \
    /**/

#define PIKA_SANITY_RANGE_(...)                                                                    \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_RANGE_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))       \
    /**/
#define PIKA_SANITY_RANGE_3(expr1, expr2, expr3)                                                   \
    PIKA_SANITY_RANGE_IMPL(::pika::detail::global_fixture, expr1, expr2, expr3)
#define PIKA_SANITY_RANGE_4(strm, expr1, expr2, expr3)                                             \
    PIKA_SANITY_RANGE_IMPL(::pika::detail::fixture{strm}, expr1, expr2, expr3)

#define PIKA_SANITY_RANGE_IMPL(fixture, expr1, expr2, expr3)                                       \
    fixture.check_range(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                          \
        ::pika::detail::counter_sanity, expr1, expr2, expr3,                                       \
        "sanity check '" PIKA_PP_STRINGIZE(expr2) " <= " PIKA_PP_STRINGIZE(                        \
            expr1) " <= " PIKA_PP_STRINGIZE(expr3) "'")

////////////////////////////////////////////////////////////////////////////////
#define PIKA_SANITY_EQ_MSG(...)                                                                    \
    PIKA_SANITY_EQ_MSG_(__VA_ARGS__)                                                               \
    /**/

#define PIKA_SANITY_EQ_MSG_(...)                                                                   \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_SANITY_EQ_MSG_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))      \
    /**/
#define PIKA_SANITY_EQ_MSG_3(expr1, expr2, msg)                                                    \
    PIKA_SANITY_EQ_MSG_IMPL(::pika::detail::global_fixture, expr1, expr2, msg)
#define PIKA_SANITY_EQ_MSG_4(strm, expr1, expr2, msg)                                              \
    PIKA_SANITY_EQ_MSG_IMPL(::pika::detail::fixture{strm}, expr1, expr2, msg)

#define PIKA_SANITY_EQ_MSG_IMPL(fixture, expr1, expr2, msg)                                        \
    fixture.check_equal(__FILE__, __LINE__, PIKA_ASSERT_CURRENT_FUNCTION,                          \
        ::pika::detail::counter_sanity, expr1, expr2, msg)

////////////////////////////////////////////////////////////////////////////////
#define PIKA_TEST_THROW(...)                                                                       \
    PIKA_TEST_THROW_(__VA_ARGS__)                                                                  \
    /**/

#define PIKA_TEST_THROW_(...)                                                                      \
    PIKA_PP_EXPAND(PIKA_PP_CAT(PIKA_TEST_THROW_, PIKA_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))         \
    /**/
#define PIKA_TEST_THROW_2(expression, exception)                                                   \
    PIKA_TEST_THROW_IMPL(::pika::detail::global_fixture, expression, exception)
#define PIKA_TEST_THROW_3(strm, expression, exception)                                             \
    PIKA_TEST_THROW_IMPL(::pika::detail::fixture{strm}, expression, exception)

#define PIKA_TEST_THROW_IMPL(fixture, expression, exception)                                       \
    {                                                                                              \
        bool caught_exception = false;                                                             \
        try                                                                                        \
        {                                                                                          \
            expression;                                                                            \
            PIKA_TEST_MSG_IMPL(fixture, false, "expected exception not thrown");                   \
        }                                                                                          \
        catch (exception&)                                                                         \
        {                                                                                          \
            caught_exception = true;                                                               \
        }                                                                                          \
        catch (...)                                                                                \
        {                                                                                          \
            PIKA_TEST_MSG_IMPL(fixture, false, "unexpected exception caught");                     \
        }                                                                                          \
        PIKA_TEST_IMPL(fixture, caught_exception);                                                 \
    }                                                                                              \
    /**/
