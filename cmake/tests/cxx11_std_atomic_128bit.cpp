////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <atomic>
#include <cstdint>

template <typename T>
void test_atomic()
{
    std::atomic<T> a;

    a.store(T{});

    {
        [[maybe_unused]] T i = a.load();
    }

    {
        [[maybe_unused]] T i = a.exchange(T{});
    }

    {
        T expected{};
        [[maybe_unused]] bool b = a.compare_exchange_weak(expected, T{});
    }

    {
        T expected{};
        [[maybe_unused]] bool b = a.compare_exchange_strong(expected, T{});
    }
}

struct uint128_type
{
    std::uint64_t left;
    std::uint64_t right;
};

int main() { test_atomic<uint128_type>(); }
