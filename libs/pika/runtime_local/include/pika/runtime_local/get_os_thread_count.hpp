//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/runtime_local/get_os_thread_count.hpp

#pragma once

#include <pika/local/config.hpp>

#include <cstddef>

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of OS-threads running in the runtime instance
    ///        the current pika-thread is associated with.
    PIKA_EXPORT std::size_t get_os_thread_count();

    namespace threads {
        class executor;
    }

    /// \brief Return the number of worker OS- threads used by the given
    ///        executor to execute pika threads
    ///
    /// This function returns the number of cores used to execute pika
    /// threads for the given executor. If the function is called while no pika
    /// runtime system is active, it will return zero. If the executor is not
    /// valid, this function will fall back to retrieving the number of OS
    /// threads used by pika.
    ///
    /// \param exec [in] The executor to be used.
    PIKA_EXPORT std::size_t get_os_thread_count(
        threads::executor const& exec);
}    // namespace pika
