//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/executors/thread_pool_executor.hpp>

namespace pika { namespace parallel { namespace execution {
    using current_executor = parallel::execution::thread_pool_executor;
}}}    // namespace pika::parallel::execution

namespace pika { namespace threads {
    ///  Returns a reference to the executor which was used to create
    /// the given thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a pika#exception with an error code of
    ///         \a pika#yield_aborted if it is signaled with \a wait_aborted.
    ///         If called outside of a pika-thread, this function will throw
    ///         an \a pika#exception with an error code of \a pika::null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a pika#exception with an error code of
    ///         \a pika#invalid_status.
    ///
    PIKA_EXPORT parallel::execution::current_executor get_executor(
        thread_id_type const& id, error_code& ec = throws);
}}    // namespace pika::threads

namespace pika { namespace this_thread {
    /// Returns a reference to the executor which was used to create the current
    /// thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a pika#exception with an error code of
    ///         \a pika#yield_aborted if it is signaled with \a wait_aborted.
    ///         If called outside of a pika-thread, this function will throw
    ///         an \a pika#exception with an error code of \a pika::null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a pika#exception with an error code of
    ///         \a pika#invalid_status.
    ///
    PIKA_EXPORT parallel::execution::current_executor get_executor(
        error_code& ec = throws);
}}    // namespace pika::this_thread
