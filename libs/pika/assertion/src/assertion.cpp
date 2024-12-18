//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>

#include <iostream>
#include <string>

namespace pika::detail {
    assertion_handler_type& get_handler()
    {
        static assertion_handler_type handler = nullptr;
        return handler;
    }

    void set_assertion_handler(assertion_handler_type handler)
    {
        if (detail::get_handler() == nullptr) { detail::get_handler() = handler; }
    }

    void handle_assert(
        source_location const& loc, char const* expr, std::string const& msg) noexcept
    {
        if (get_handler() == nullptr)
        {
            std::cerr << loc << ": Assertion '" << expr << "' failed";
            if (!msg.empty()) { std::cerr << " (" << msg << ")\n"; }
            else { std::cerr << '\n'; }
            std::abort();
        }
        get_handler()(loc, expr, msg);
    }
}    // namespace pika::detail
