//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(PIKA_HAVE_MODULE)
#include <algorithm>
#include <cctype>
#include <string>
#endif

namespace pika::detail {
    template <typename CharT, class Traits, class Alloc>
    void to_lower(std::basic_string<CharT, Traits, Alloc>& s)
    {
        std::transform(
            std::begin(s), std::end(s), std::begin(s), [](int c) { return std::tolower(c); });
    }

    template <typename CharT, class Traits, class Alloc>
    void to_upper(std::basic_string<CharT, Traits, Alloc>& s)
    {
        std::transform(
            std::begin(s), std::end(s), std::begin(s), [](int c) { return std::toupper(c); });
    }
}    // namespace pika::detail
