////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/string_util/from_string.hpp>
#include <pika/string_util/trim.hpp>
#include <pika/type_support/unused.hpp>

#include <fmt/format.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>

// clang-format off
#if defined(PIKA_HAVE_MORE_THAN_64_THREADS) ||                                  \
    (defined(PIKA_HAVE_MAX_CPU_COUNT) && PIKA_HAVE_MAX_CPU_COUNT > 64)
#  if defined(PIKA_HAVE_MAX_CPU_COUNT)
#    include <bitset>
#  else
#    include <boost/dynamic_bitset.hpp>
#  endif
#endif
// clang-format on

namespace pika::threads::detail {
    /// \cond NOINTERNAL
#if !defined(PIKA_HAVE_MORE_THAN_64_THREADS) ||                                                    \
    (defined(PIKA_HAVE_MAX_CPU_COUNT) && PIKA_HAVE_MAX_CPU_COUNT <= 64)
    using mask_type = std::uint64_t;
    using mask_cref_type = std::uint64_t;

    inline std::uint64_t bits(std::size_t idx)
    {
        PIKA_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        return std::uint64_t(1) << idx;
    }

    inline bool any(mask_cref_type mask) { return mask != 0; }

    inline mask_type not_(mask_cref_type mask) { return ~mask; }

    inline bool test(mask_cref_type mask, std::size_t idx)
    {
        PIKA_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        return (bits(idx) & mask) != 0;
    }

    inline void set(mask_type& mask, std::size_t idx)
    {
        PIKA_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        mask |= bits(idx);
    }

    inline void unset(mask_type& mask, std::size_t idx)
    {
        PIKA_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        mask &= not_(bits(idx));
    }

    inline std::size_t mask_size(mask_cref_type /*mask*/) { return CHAR_BIT * sizeof(mask_type); }

    inline void resize(mask_type& /*mask*/, std::size_t s)
    {
        PIKA_ASSERT(s <= CHAR_BIT * sizeof(mask_type));
        PIKA_UNUSED(s);
    }

    inline std::size_t find_first(mask_cref_type mask)
    {
        if (mask)
        {
            std::size_t c = 0;    // Will count mask's trailing zero bits.

            // Set mask's trailing 0s to 1s and zero rest.
            mask = (mask ^ (mask - 1)) >> 1;
            for (/**/; mask; ++c) mask >>= 1;

            return c;
        }
        return ~std::size_t(0);
    }

    inline bool equal(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return lhs == rhs;
    }

    // return true if at least one of the masks has a bit set
    inline bool bit_or(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs | rhs) != 0;
    }

    // return true if at least one bit is set in both masks
    inline bool bit_and(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs & rhs) != 0;
    }

    // returns the number of bits set
    // taken from https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    inline std::size_t count(mask_cref_type mask)
    {
        std::size_t c;    // c accumulates the total bits set in v
        for (c = 0; mask; c++)
        {
            mask &= mask - 1;    // clear the least significant bit set
        }
        return c;
    }

    inline void reset(mask_type& mask) { mask = 0ull; }

    // clang-format off
#else
#  if defined(PIKA_HAVE_MAX_CPU_COUNT)
    using mask_type = std::bitset<PIKA_HAVE_MAX_CPU_COUNT>;
    using mask_cref_type = std::bitset<PIKA_HAVE_MAX_CPU_COUNT> const&;
#  else
    using mask_type = boost::dynamic_bitset<std::uint64_t>;
    using mask_cref_type = boost::dynamic_bitset<std::uint64_t> const&;
#  endif
    // clang-format on

    inline bool any(mask_cref_type mask) { return mask.any(); }

    inline mask_type not_(mask_cref_type mask) { return ~mask; }

    inline bool test(mask_cref_type mask, std::size_t idx) { return mask.test(idx); }

    inline void set(mask_type& mask, std::size_t idx) { mask.set(idx); }

    inline void unset(mask_type& mask, std::size_t idx) { mask.set(idx, 0); }

    inline std::size_t mask_size(mask_cref_type mask) { return mask.size(); }

    // clang-format off
    inline void resize(mask_type& mask, std::size_t s)
    {
#  if defined(PIKA_HAVE_MAX_CPU_COUNT)
        PIKA_ASSERT(s <= mask.size());
        PIKA_UNUSED(mask);
        PIKA_UNUSED(s);
#  else
        return mask.resize(s);
#  endif
    }

    inline std::size_t find_first(mask_cref_type mask)
    {
#  if defined(PIKA_HAVE_MAX_CPU_COUNT)
        if (mask.any())
        {
            for (std::size_t i = 0; i != PIKA_HAVE_MAX_CPU_COUNT; ++i)
            {
                if (mask[i])
                    return i;
            }
        }
        return ~std::size_t(0);
#  else
        return mask.find_first();
#  endif
    }
    // clang-format on

    inline bool equal(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return lhs == rhs;
    }

    // return true if at least one of the masks has a bit set
    inline bool bit_or(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs | rhs).any();
    }

    // return true if at least one bit is set in both masks
    inline bool bit_and(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs & rhs).any();
    }

    // returns the number of bits set
    inline std::size_t count(mask_cref_type mask) { return mask.count(); }

    inline void reset(mask_type& mask) { mask.reset(); }

#endif

    PIKA_EXPORT std::string to_string(mask_cref_type);
    /// \endcond
}    // namespace pika::threads::detail

namespace pika::detail {
    template <>
    struct from_string_impl<pika::threads::detail::mask_type, void>
    {
        template <typename Char>
        static void
        call(std::basic_string<Char> const& value, pika::threads::detail::mask_type& target)
        {
            // Trim whitespace from beginning and end
            std::basic_string<Char> value_trimmed = value;
            pika::detail::trim(value_trimmed);

            if (value_trimmed.size() < 3)
            {
                throw std::out_of_range(
                    fmt::format("from_string<mask_type>: hexadecimal string (\"{}\"), expecting a "
                                "prefix of 0x and at least one digit",
                        value_trimmed));
            }

            if (value_trimmed.find("0x") != 0)
            {
                throw std::out_of_range(fmt::format("from_string<mask_type>: hexadecimal string "
                                                    "(\"{}\") does not start with \"0x\"",
                    value_trimmed));
            }

            // Convert a potentially hexadecimal character to an integer (mask) between 0 and 15
            constexpr auto const to_mask = [](unsigned char const c) {
                if (48 <= c && c < 58) { return c - 48; }
                else if (auto const c_lower = std::tolower(c); 97 <= c_lower && c_lower < 103)
                {
                    return c_lower - 87;
                }

                throw std::out_of_range(fmt::format(
                    "from_string<mask_type>: got invalid hexadecimal character (\"{}\")", c));
            };

            pika::threads::detail::reset(target);
            pika::threads::detail::resize(target, 0);

            for (auto begin = value_trimmed.begin() + 2; begin != value_trimmed.cend(); ++begin)
            {
                // Each character read represents 4 bits so we make space for those bytes
#if !defined(PIKA_HAVE_MAX_CPU_COUNT)
                pika::threads::detail::resize(target, pika::threads::detail::mask_size(target) + 4);
#endif
                target <<= 4;

                // Store the current 4 bits into a mask of the same size as the target
#if defined(PIKA_HAVE_MAX_CPU_COUNT)
                pika::threads::detail::mask_type cur(to_mask(*begin));
#else
                pika::threads::detail::mask_type cur(
                    pika::threads::detail::mask_size(target), to_mask(*begin));
#endif

                // Add the newly read bits to the mask
                target |= cur;
            }
        }
    };
}    // namespace pika::detail
