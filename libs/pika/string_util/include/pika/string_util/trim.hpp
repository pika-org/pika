//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(PIKA_HAVE_MODULE)
#include <algorithm>
#include <cstddef>
#include <string>
#endif

namespace pika::detail {
    template <typename CharT, class Traits, class Alloc>
    void trim(std::basic_string<CharT, Traits, Alloc>& s)
    {
        // When using the pre-C++11 ABI in libstdc++, basic_string::erase
        // does not have an overload taking different begin and end iterators.
#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI == 0
        auto first = std::find_if_not(std::begin(s), std::end(s),
#else
        auto first = std::find_if_not(std::cbegin(s), std::cend(s),
#endif
            [](int c) { return std::isspace(c); });
        s.erase(std::begin(s), first);

#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI == 0
        auto last = std::find_if_not(std::rbegin(s), std::rend(s),
#else
        auto last = std::find_if_not(std::crbegin(s), std::crend(s),
#endif
            [](int c) { return std::isspace(c); });
        s.erase(last.base(), std::end(s));
    }

    template <typename CharT, class Traits, class Alloc>
    std::basic_string<CharT, Traits, Alloc>
    trim_copy(std::basic_string<CharT, Traits, Alloc> const& s)
    {
        auto t = s;
        trim(t);

        return t;
    }
}    // namespace pika::detail
