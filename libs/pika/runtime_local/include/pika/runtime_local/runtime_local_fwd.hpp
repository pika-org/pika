//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file runtime_local_fwd.hpp

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/runtime_configuration.hpp>
#include <pika/runtime_local/config_entry.hpp>
#include <pika/runtime_local/detail/runtime_local_fwd.hpp>
#include <pika/runtime_local/get_locality_id.hpp>
#include <pika/runtime_local/get_num_all_localities.hpp>
#include <pika/runtime_local/get_os_thread_count.hpp>
#include <pika/runtime_local/get_thread_name.hpp>
#include <pika/runtime_local/get_worker_thread_num.hpp>
#include <pika/runtime_local/os_thread_type.hpp>
#include <pika/runtime_local/report_error.hpp>
#include <pika/runtime_local/shutdown_function.hpp>
#include <pika/runtime_local/startup_function.hpp>
#include <pika/runtime_local/thread_hooks.hpp>
#include <pika/runtime_local/thread_pool_helpers.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threadmanager/threadmanager_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace pika {
    /// Register the current kernel thread with pika, this should be done once
    /// for each external OS-thread intended to invoke pika functionality.
    /// Calling this function more than once will return false.
    PIKA_EXPORT bool register_thread(
        runtime* rt, char const* name, error_code& ec = throws);

    /// Unregister the thread from pika, this should be done once in
    /// the end before the external thread exists.
    PIKA_EXPORT void unregister_thread(runtime* rt);

    /// Access data for a given OS thread that was previously registered by
    /// \a register_thread. This function must be called from a thread that was
    /// previously registered with the runtime.
    PIKA_EXPORT runtime_local::os_thread_data get_os_thread_data(
        std::string const& label);

    /// Enumerate all OS threads that have registered with the runtime.
    PIKA_EXPORT bool enumerate_os_threads(
        util::function_nonser<bool(os_thread_data const&)> const& f);

    /// Return the runtime instance number associated with the runtime instance
    /// the current thread is running in.
    PIKA_EXPORT std::size_t get_runtime_instance_number();

    /// Register a function to be called during system shutdown
    PIKA_EXPORT bool register_on_exit(
        util::function_nonser<void()> const&);

    /// \cond NOINTERNAL
    namespace util {
        /// \brief Expand INI variables in a string
        PIKA_EXPORT std::string expand(std::string const& expand);

        /// \brief Expand INI variables in a string
        PIKA_EXPORT void expand(std::string& expand);
    }    // namespace util

    ///////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT bool is_scheduler_numa_sensitive();

    ///////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT pika::util::runtime_configuration const& get_config();

    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently being started.
    ///
    /// This function returns whether the runtime system is currently being
    /// started or not, e.g. whether the current state of the runtime system is
    /// \a pika::state_startup
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
    /// \a pika::state_running
    ///
    /// \note   This function needs to be executed on a pika-thread. It will
    ///         return false otherwise.
    PIKA_EXPORT bool is_running();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently stopped.
    ///
    /// This function returns whether the runtime system is currently stopped
    /// or not, e.g.  whether the current state of the runtime system is
    /// \a pika::state_stopped
    ///
    /// \note   This function needs to be executed on a pika-thread. It will
    ///         return false otherwise.
    PIKA_EXPORT bool is_stopped();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Test whether the runtime system is currently being shut down.
    ///
    /// This function returns whether the runtime system is currently being
    /// shut down or not, e.g.  whether the current state of the runtime system
    /// is \a pika::state_stopped or \a pika::state_shutdown
    ///
    /// \note   This function needs to be executed on a pika-thread. It will
    ///         return false otherwise.
    PIKA_EXPORT bool is_stopped_or_shutting_down();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of worker OS- threads used to execute pika
    ///        threads
    ///
    /// This function returns the number of OS-threads used to execute pika
    /// threads. If the function is called while no pika runtime system is active,
    /// it will return zero.
    PIKA_EXPORT std::size_t get_num_worker_threads();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the system uptime measure on the thread executing this call.
    ///
    /// This function returns the system uptime measured in nanoseconds for the
    /// thread executing this call. If the function is called while no pika
    /// runtime system is active, it will return zero.
    PIKA_EXPORT std::uint64_t get_system_uptime();

    namespace threads {

        /// \cond NOINTERNAL
        // The function get_thread_manager returns a reference to the
        // current thread manager.
        PIKA_EXPORT threadmanager& get_thread_manager();
        /// \endcond

        /// \cond NOINTERNAL
        /// Reset internal (round robin) thread distribution scheme
        PIKA_EXPORT void reset_thread_distribution();

        /// Set the new scheduler mode
        PIKA_EXPORT void set_scheduler_mode(
            threads::policies::scheduler_mode new_mode);

        /// Add the given flags to the scheduler mode
        PIKA_EXPORT void add_scheduler_mode(
            threads::policies::scheduler_mode to_add);

        /// Add/remove the given flags to the scheduler mode
        PIKA_EXPORT void add_remove_scheduler_mode(
            threads::policies::scheduler_mode to_add,
            threads::policies::scheduler_mode to_remove);

        /// Remove the given flags from the scheduler mode
        PIKA_EXPORT void remove_scheduler_mode(
            threads::policies::scheduler_mode to_remove);

        /// Get the global topology instance
        PIKA_EXPORT topology const& get_topology();
        /// \endcond
    }    // namespace threads

    namespace detail {
        PIKA_EXPORT void on_exit() noexcept;
        PIKA_EXPORT void on_abort(int signal) noexcept;
        PIKA_EXPORT void handle_print_bind(std::size_t num_threads);
    }    // namespace detail
}    // namespace pika
