//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/program_options.hpp>
#include <pika/testing.hpp>

#include <optional>
#include <string>
#include <vector>

namespace po = pika::program_options;

std::vector<std::string> sv(char const* array[], unsigned size)
{
    std::vector<std::string> r;
    for (unsigned i = 0; i < size; ++i) r.emplace_back(array[i]);
    return r;
}

void test_optional()
{
    std::optional<int> foo, bar, baz;

    po::options_description desc;
    // clang-format off
    desc.add_options()
        ("foo,f", po::value(&foo), "")
        ("bar,b", po::value(&bar), "")
        ("baz,z", po::value(&baz), "")
        ;
    // clang-format on

    char const* cmdline1_[] = {"--foo=12", "--bar", "1"};
    std::vector<std::string> cmdline1 = sv(cmdline1_, sizeof(cmdline1_) / sizeof(char const*));

    po::variables_map vm;
    po::store(po::command_line_parser(cmdline1).options(desc).run(), vm);
    po::notify(vm);

    PIKA_TEST(!!foo);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    PIKA_TEST_EQ(*foo, 12);

    PIKA_TEST(!!bar);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    PIKA_TEST_EQ(*bar, 1);

    PIKA_TEST(!baz);
}

int main(int, char*[])
{
    test_optional();
    return 0;
}
