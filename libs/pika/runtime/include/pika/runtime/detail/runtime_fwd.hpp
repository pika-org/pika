//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/runtime/runtime_fwd.hpp

#pragma once

#include <pika/config.hpp>

namespace pika {

    class PIKA_EXPORT runtime;

    /// The function \a get_runtime returns a reference to the (thread
    /// specific) runtime instance.
    PIKA_EXPORT runtime& get_runtime();
    PIKA_EXPORT runtime*& get_runtime_ptr();
}    // namespace pika
