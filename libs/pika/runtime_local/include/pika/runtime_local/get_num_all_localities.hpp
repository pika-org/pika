//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_num_all_localities.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/futures/future_fwd.hpp>
#include <pika/modules/errors.hpp>

#include <cstdint>

namespace pika {

    /// \brief Return the number of localities which were registered at startup
    ///        for the running application.
    ///
    /// The function \a get_initial_num_localities returns the number of localities
    /// which were connected to the console at application startup.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a pika::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           pika::exception.
    ///
    /// \see      \a pika::find_all_localities, \a pika::get_num_localities
    PIKA_EXPORT std::uint32_t get_initial_num_localities();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Asynchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities asynchronously returns the
    /// number of localities currently connected to the console. The returned
    /// future represents the actual result.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an pika-thread. It will return 0 otherwise.
    ///
    /// \see      \a pika::find_all_localities, \a pika::get_num_localities
    PIKA_EXPORT pika::future<std::uint32_t> get_num_localities();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a pika#throws the function will throw
    ///           on error instead.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an pika-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a pika::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           pika::exception.
    ///
    /// \see      \a pika::find_all_localities, \a pika::get_num_localities
    PIKA_EXPORT std::uint32_t get_num_localities(
        launch::sync_policy, error_code& ec = throws);
}    // namespace pika
