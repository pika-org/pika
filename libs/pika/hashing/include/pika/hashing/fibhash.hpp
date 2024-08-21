// Copyright (c) 2018 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This code is based on the article found here:
// https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <cstdint>
#include <cstdlib>
#endif

namespace pika::detail {
    template <std::uint64_t N>
    struct fibhash_helper;

    template <>
    struct fibhash_helper<0>
    {
        static constexpr int log2 = -1;
    };

    template <std::uint64_t N>
    struct fibhash_helper
    {
        static constexpr std::uint64_t log2 = fibhash_helper<(N >> 1)>::log2 + 1;
        static constexpr std::uint64_t shift_amount = 64 - log2;
    };

    inline constexpr std::uint64_t golden_ratio = 11400714819323198485llu;

    // This function calculates the hash based on a multiplicative Fibonacci
    // scheme
    template <std::uint64_t N>
    constexpr std::uint64_t fibhash(std::uint64_t i) noexcept
    {
        using helper = fibhash_helper<N>;
        static_assert(N != 0, "This algorithm only works with N != 0");
        static_assert((1 << helper::log2) == N, "N must be a power of two");    // -V104
        return (detail::golden_ratio * (i ^ (i >> helper::shift_amount))) >> helper::shift_amount;
    }
}    // namespace pika::detail
