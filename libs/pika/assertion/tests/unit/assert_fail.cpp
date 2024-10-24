//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/testing.hpp>

#include <string>

[[noreturn]] void assertion_handler(
    pika::detail::source_location const&, char const*, std::string const&)
{
    PIKA_TEST(true);
    std::exit(1);
}

int main()
{
    // We set a custom assertion handler because the default one aborts, which
    // ctest considers a fatal error, even if WILL_FAIL is set to true.
    pika::detail::set_assertion_handler(&assertion_handler);
    PIKA_ASSERT(false);

    PIKA_TEST(true);
}
