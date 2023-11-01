//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/command_line_handling/get_env_var_as.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int pika_main()
{
    float f = pika::detail::get_env_var_as<float>("float", 0.0);
    PIKA_TEST_EQ(f, 3.1415f);

    double d = pika::detail::get_env_var_as<double>("double", 0.0);
    PIKA_TEST_EQ(d, 6.28);

    std::int32_t i32 = pika::detail::get_env_var_as<std::int32_t>("int32", 0);
    PIKA_TEST_EQ(i32, -65536);

    std::int64_t i64 = pika::detail::get_env_var_as<std::int64_t>("int64", 0);
    PIKA_TEST_EQ(i64, -123465536);

    std::uint32_t u32 = pika::detail::get_env_var_as<std::uint32_t>("uint32", 0);
    PIKA_TEST_EQ(u32, 65536u);

    std::uint64_t u64 = pika::detail::get_env_var_as<std::uint64_t>("uint64", 0);
    PIKA_TEST_EQ(u64, 123465536u);

    std::string s = pika::detail::get_env_var_as<std::string>("string", "wrong");
    PIKA_TEST_EQ(s, std::string("hello-world"));

    pika::finalize();
    return EXIT_SUCCESS;
}
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    putenv(const_cast<char*>("float=3.1415"));
    putenv(const_cast<char*>("double=6.28"));
    putenv(const_cast<char*>("int32=-65536"));
    putenv(const_cast<char*>("int64=-123465536"));
    putenv(const_cast<char*>("uint32=65536"));
    putenv(const_cast<char*>("uint64=123465536"));
    putenv(const_cast<char*>("string=hello-world"));

    PIKA_TEST_EQ(pika::init(pika_main, argc, argv), 0);
    return 0;
}
