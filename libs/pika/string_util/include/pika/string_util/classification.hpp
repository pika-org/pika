//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if !defined(PIKA_HAVE_MODULE)
#include <cctype>
#include <string>
#endif

namespace pika::detail {
    template <typename CharT, typename Traits, typename Allocator>
    struct is_any_of_pred
    {
        bool operator()(int c) const noexcept { return chars.find(c) != std::string::npos; }

        std::basic_string<CharT, Traits, Allocator> chars;
    };

    template <typename CharT, typename Traits, typename Allocator>
    is_any_of_pred<CharT, Traits, Allocator>
    is_any_of(std::basic_string<CharT, Traits, Allocator> const& chars)
    {
        return is_any_of_pred<CharT, Traits, Allocator>{chars};
    }

    inline auto is_any_of(char const* chars)
    {
        return is_any_of_pred<char, std::char_traits<char>, std::allocator<char>>{
            std::string{chars}};
    }

}    // namespace pika::detail
