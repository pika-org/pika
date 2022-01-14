//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_thread_name.hpp

#pragma once

#include <pika/local/config.hpp>

#include <string>

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the name of the calling thread.
    ///
    /// This function returns the name of the calling thread. This name uniquely
    /// identifies the thread in the context of pika. If the function is called
    /// while no pika runtime system is active, the result will be "<unknown>".
    PIKA_EXPORT std::string get_thread_name();
}    // namespace pika
