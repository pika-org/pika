//  Copyright (c) 2018 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/modules/format.hpp>
#include <pika/modules/testing.hpp>

#include <ctime>
#include <string>
#include <vector>

int main()
{
    using pika::util::format;
    {
        PIKA_TEST_EQ((format("Hello")), "Hello");
        PIKA_TEST_EQ((format("Hello, {}!", "world")), "Hello, world!");
        PIKA_TEST_EQ((format("The number is {}", 1)), "The number is 1");
    }

    {
        PIKA_TEST_EQ((format("{} {}", 1, 2)), "1 2");
        PIKA_TEST_EQ((format("{} {1}", 1, 2)), "1 1");
        PIKA_TEST_EQ((format("{2} {}", 1, 2)), "2 2");
        PIKA_TEST_EQ((format("{2} {1}", 1, 2)), "2 1");

        PIKA_TEST_EQ((format("{:}", 42)), "42");
        PIKA_TEST_EQ((format("{:04}", 42)), "0042");
        PIKA_TEST_EQ((format("{2:04}", 42, 43)), "0043");

        PIKA_TEST_EQ((format("{:x}", 42)), "2a");
        PIKA_TEST_EQ((format("{:04x}", 42)), "002a");
        PIKA_TEST_EQ((format("{2:04x}", 42, 43)), "002b");

        PIKA_TEST_EQ((format("{:#x}", 42)), "0x2a");
        PIKA_TEST_EQ((format("{:#06x}", 42)), "0x002a");
        PIKA_TEST_EQ((format("{2:#06x}", 42, 43)), "0x002b");
    }

    {
        PIKA_TEST_EQ((format("{} {}", true, false)), "1 0");
    }

    {
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        char buffer[64] = {};
        std::strftime(buffer, 64, "%c", &tm);
        PIKA_TEST_EQ((format("{}", tm)), buffer);

        std::strftime(buffer, 64, "%A %c", &tm);
        PIKA_TEST_EQ((format("{:%A %c}", tm)), buffer);
    }

    {
        using pika::util::format_join;
        std::vector<int> const vs = {42, 43};
        PIKA_TEST_EQ((format("{}", format_join(vs, ""))), "4243");
        PIKA_TEST_EQ((format("{}", format_join(vs, ","))), "42,43");
        PIKA_TEST_EQ((format("{:x}", format_join(vs, ""))), "2a2b");
        PIKA_TEST_EQ((format("{:04x}", format_join(vs, ","))), "002a,002b");
    }

    {
        PIKA_TEST_EQ((format("{{ {}", 1)), "{ 1");
        PIKA_TEST_EQ((format("}} {}", 1)), "} 1");
        PIKA_TEST_EQ((format("{{}} {}", 1)), "{} 1");
        PIKA_TEST_EQ((format("{} {{}}", 1)), "1 {}");
        PIKA_TEST_EQ((format("{} {{", 1)), "1 {");
        PIKA_TEST_EQ((format("{} }}", 1)), "1 }");
        PIKA_TEST_EQ((format("{{{1}}}", 2)), "{2}");
    }

    return pika::util::report_errors();
}
