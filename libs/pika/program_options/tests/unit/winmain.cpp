// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/testing.hpp>

#if defined(PIKA_WINDOWS)
# include <pika/preprocessor/cat.hpp>
# include <pika/program_options/parsers.hpp>

# include <cctype>
# include <cstdlib>
# include <iostream>
# include <string>
# include <vector>

using namespace pika::program_options;
using namespace std;

void check_equal(std::vector<string> const& actual, char const** expected, int n)
{
    if (actual.size() != n)
    {
        std::cerr << "Size mismatch between expected and actual data\n";
        abort();
    }
    for (int i = 0; i < n; ++i)
    {
        if (actual[i] != expected[i])
        {
            std::cerr << "Unexpected content\n";
            abort();
        }
    }
}

# define COMMA ,
# define TEST(input, expected)                                                                     \
     char const* PIKA_PP_CAT(e, __LINE__)[] = expected;                                            \
     vector<string> PIKA_PP_CAT(v, __LINE__) = split_winmain(input);                               \
     check_equal(PIKA_PP_CAT(v, __LINE__), PIKA_PP_CAT(e, __LINE__),                               \
         sizeof(PIKA_PP_CAT(e, __LINE__)) / sizeof(char*));                                        \
     /**/

void test_winmain()
{
    // The following expectations were obtained in Win2000 shell:
    TEST("1 ", {"1"});
    TEST("1\"2\" ", {"12"});
    TEST("1\"2  ", {"12  "});
    TEST("1\"\\\"2\" ", {"1\"2"});
    TEST("\"1\" \"2\" ", {"1" COMMA "2"});
    TEST("1\\\" ", {"1\""});
    TEST("1\\\\\" ", {"1\\ "});
    TEST("1\\\\\\\" ", {"1\\\""});
    TEST("1\\\\\\\\\" ", {"1\\\\ "});

    TEST("1\" 1 ", {"1 1 "});
    TEST("1\\\" 1 ", {"1\"" COMMA "1"});
    TEST("1\\1 ", {"1\\1"});
    TEST("1\\\\1 ", {"1\\\\1"});
}

int main(int, char*[])
{
    test_winmain();
    return 0;
}
#else
int main(int, char*[])
{
    // There is nothing to test if not on Windows
    PIKA_TEST(true);
    return 0;
}
#endif
