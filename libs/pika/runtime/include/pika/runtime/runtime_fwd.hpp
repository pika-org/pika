//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file runtime_fwd.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/runtime_configuration.hpp>
#include <pika/runtime/config_entry.hpp>
#include <pika/runtime/get_os_thread_count.hpp>
#include <pika/runtime/get_thread_name.hpp>
#include <pika/runtime/get_worker_thread_num.hpp>
#include <pika/runtime/report_error.hpp>
#include <pika/runtime/shutdown_function.hpp>
#include <pika/runtime/startup_function.hpp>
#include <pika/runtime/thread_hooks.hpp>
#include <pika/runtime/thread_pool_helpers.hpp>
#include <pika/thread_manager/thread_manager_fwd.hpp>
#include <pika/threading_base/scheduler_base.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace pika::detail {

    class PIKA_EXPORT runtime;

    /// The function \a get_runtime returns a reference to the (thread
    /// specific) runtime instance.
    PIKA_EXPORT runtime& get_runtime();
    PIKA_EXPORT runtime*& get_runtime_ptr();

    /// Register the current kernel thread with pika, this should be done once
    /// for each external OS-thread intended to invoke pika functionality.
    /// Calling this function more than once will return false.
    PIKA_EXPORT bool register_thread(runtime* rt, char const* name, error_code& ec = throws);

    /// Unregister the thread from pika, this should be done once in
    /// the end before the external thread exists.
    PIKA_EXPORT void unregister_thread(runtime* rt);

    /// Register a function to be called during system shutdown
    PIKA_EXPORT bool register_on_exit(pika::util::detail::function<void()> const&);

    ///////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT pika::util::runtime_configuration const& get_config();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently being started.
    ///
    /// This function returns whether the runtime system is currently being
    /// started or not, e.g. whether the current state of the runtime system is
    /// \a pika:runtime_state::startup
    ///
    /// \note   This function needs to be executed on a pika-thread. It will
    ///         return false otherwise.
    PIKA_EXPORT bool is_starting();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test if pika runs in fault-tolerant mode
    ///
    /// This function returns whether the runtime system is running
    /// in fault-tolerant mode
    PIKA_EXPORT bool tolerate_node_faults();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently running.
    ///
    /// This function returns whether the runtime system is currently running
    /// or not, e.g.  whether the current state of the runtime system is
    /// \a pika:runtime_state::running
    ///
    /// \note   This function needs to be executed on a pika-thread. It will
    ///         return false otherwise.
    PIKA_EXPORT bool is_running();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently stopped.
    ///
    /// This function returns whether the runtime system is currently stopped
    /// or not, e.g.  whether the current state of the runtime system is
    /// \a pika:runtime_state::stopped
    ///
    /// \note   This function needs to be executed on a pika-thread. It will
    ///         return false otherwise.
    PIKA_EXPORT bool is_stopped();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently being shut down.
    ///
    /// This function returns whether the runtime system is currently being
    /// shut down or not, e.g.  whether the current state of the runtime system
    /// is \a pika:runtime_state::stopped or \a pika:runtime_state::shutdown
    ///
    /// \note   This function needs to be executed on a pika-thread. It will
    ///         return false otherwise.
    PIKA_EXPORT bool is_stopped_or_shutting_down();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the system uptime measure on the thread executing this call.
    ///
    /// This function returns the system uptime measured in nanoseconds for the
    /// thread executing this call. If the function is called while no pika
    /// runtime system is active, it will return zero.
    PIKA_EXPORT std::uint64_t get_system_uptime();

    /// \cond NOINTERNAL
    /// Reset internal (round robin) thread distribution scheme
    PIKA_EXPORT void reset_thread_distribution();

    /// Set the new scheduler mode
    PIKA_EXPORT void set_scheduler_mode(threads::scheduler_mode new_mode);

    /// Add the given flags to the scheduler mode
    PIKA_EXPORT void add_scheduler_mode(threads::scheduler_mode to_add);

    /// Remove the given flags from the scheduler mode
    PIKA_EXPORT void remove_scheduler_mode(threads::scheduler_mode to_remove);

    /// Get the global topology instance
    PIKA_EXPORT pika::threads::detail::topology const& get_topology();
    /// \endcond

    PIKA_EXPORT void on_exit() noexcept;
    PIKA_EXPORT void on_abort(int signal) noexcept;
    PIKA_EXPORT void handle_print_bind(std::size_t num_threads);
}    // namespace pika::detail

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of worker OS- threads used to execute pika
    ///        threads
    ///
    /// This function returns the number of OS-threads used to execute pika
    /// threads. If the function is called while no pika runtime system is active,
    /// it will return zero.
    PIKA_EXPORT std::size_t get_num_worker_threads();
}    // namespace pika
