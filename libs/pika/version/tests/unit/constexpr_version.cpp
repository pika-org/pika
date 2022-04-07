//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/version.hpp>

#include <cstdint>

template <std::uint8_t, std::uint8_t, std::uint8_t>
void test_constexpr_versions()
{
}

int main()
{
    test_constexpr_versions<pika::major_version(), pika::minor_version(),
        pika::patch_version()>();
}
