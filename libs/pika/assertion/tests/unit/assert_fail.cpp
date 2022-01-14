//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>

#include <string>

PIKA_NORETURN void assertion_handler(
    pika::assertion::source_location const&, const char*, std::string const&)
{
    std::exit(1);
}

int main()
{
    // We set a custom assertion handler because the default one aborts, which
    // ctest considers a fatal error, even if WILL_FAIL is set to true.
    pika::assertion::set_assertion_handler(&assertion_handler);
    PIKA_ASSERT(false);
}
