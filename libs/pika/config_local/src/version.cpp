////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/local/config/export_definitions.hpp>
#include <pika/local/config/version.hpp>
#include <pika/preprocessor/stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika::local {
    PIKA_EXPORT char const PIKA_CHECK_VERSION[] =
        PIKA_PP_STRINGIZE(PIKA_CHECK_VERSION);
    PIKA_EXPORT char const PIKA_CHECK_BOOST_VERSION[] =
        PIKA_PP_STRINGIZE(PIKA_CHECK_BOOST_VERSION);
}    // namespace pika::local
