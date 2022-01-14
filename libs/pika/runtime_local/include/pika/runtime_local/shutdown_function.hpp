//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file startup_function.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/unique_function.hpp>

namespace pika {
    /// The type of a function which is registered to be executed as a
    /// shutdown or pre-shutdown function.
    typedef util::unique_function_nonser<void()> shutdown_function_type;

    /// \brief Add a function to be executed by a pika thread during
    /// \a pika::finalize() but guaranteed before any shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_pre_shutdown_function
    /// are guaranteed to be executed by an pika thread during the execution of
    /// \a pika::finalize() before any of the registered shutdown functions are
    /// executed (see: \a pika::register_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an pika thread as
    ///           a pre-shutdown function.
    ///
    /// \note If this function is called while the pre-shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a pika::register_shutdown_function()
    PIKA_EXPORT void register_pre_shutdown_function(
        shutdown_function_type f);

    /// \brief Add a function to be executed by a pika thread during
    /// \a pika::finalize() but guaranteed after any pre-shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_shutdown_function
    /// are guaranteed to be executed by an pika thread during the execution of
    /// \a pika::finalize() after any of the registered pre-shutdown functions
    /// are executed (see: \a pika::register_pre_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an pika thread as
    ///           a shutdown function.
    ///
    /// \note If this function is called while the shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a pika::register_pre_shutdown_function()
    PIKA_EXPORT void register_shutdown_function(shutdown_function_type f);
}    // namespace pika
