//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/modules/concurrency.hpp>
#include <pika/modules/execution_base.hpp>
#include <pika/modules/memory.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/runtime/thread_pool_helpers.hpp>
#include <pika/synchronization/counting_semaphore.hpp>
#include <pika/type_support/to_underlying.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pika::mpi::experimental {

    /// mode to control installation of pika mpi error -> exception handler
    enum exception_mode
    {
        no_handler = 0,
        install_handler,
    };

    /// This enumeration describes polling pool creation modes,
    /// the user may request a dedicated pool that can be used by pika::mpi
    /// MPI pool completion flags are passed on the command line or via env vars
    /// the default mode is pika_decides: if running on one rank or a single thread
    /// per rank, creation of the pool will be
    enum polling_pool_creation_mode
    {
        /// if the completion mode requires it, a pool will be created at startup
        mode_pika_decides = 0,
        /// overrides command line flags/env vars - enables creation of a polling pool
        mode_force_create = 1,
    };

    namespace detail {
        using request_callback_function_type = pika::util::detail::unique_function<void(int)>;

        // -----------------------------------------------------------------
        /// Adds an mpi request to the internal queues for polling/handling
        PIKA_EXPORT bool add_request_callback(request_callback_function_type&&, MPI_Request);

        // -----------------------------------------------------------------
        /// MPI Error handling/interception
        /// an MPI error handling type that we can use to intercept
        /// MPI errors if we enable the error handler
        PIKA_EXPORT extern MPI_Errhandler pika_mpi_errhandler;

        /// function that converts an MPI error into an exception
        PIKA_EXPORT void pika_MPI_Handler(MPI_Comm*, int* /*errorcode*/, ...);

        /// set an error handler for communicators that will be called
        /// on any error instead of the default behavior of program termination
        PIKA_EXPORT void set_error_handler();

        /// utility function to avoid duplication in eager check locations
        PIKA_EXPORT bool poll_request(MPI_Request /*req*/);

        // -----------------------------------------------------------------
        // handling of requests and continuations
        /// flags that control how mpi continuations are handled
        enum class handler_method : std::uint32_t
        {
            /// this bit enables the inline invocation of the mpi request, when set
            /// the calling thread performs the mpi operation, when unset, a transfer
            /// is made so that the invocation happens on a new task that
            /// would normally be on a dedicated pool if/when it exists
            request_inline = 0b0000'0001,    // 1

            /// this bit enables the inline execution of the completion handler for the
            /// request, when unset a transfer is made to move the completion handler
            /// from the polling thread onto a new one
            completion_inline = 0b0000'0010,    // 2

            /// this bit enables the use of a high priority task flag
            /// 1) requests are boosted to high priority if they are passed the the mpi-pool
            ///    to ensure they execute before other polling tasks (reduce latency)
            /// 2) completions are boosted to high priority when sent to the main thread pool
            ///    so that the continuation is executed as quickly as possible
            high_priority = 0b0000'0100,    // 4

            /// 3 bits control the handler method,
            method_mask = 0b0011'1000,    // 56

            /// Methods supported for dispatching continuations:
            ///
            /// * yield_while : after a task submits an mpi_request, it yields (goes into
            /// suspension) and then polls the request every time it awakes and repeatedly yields
            /// until the request is ready, when it resumes executing the continuation
            ///
            /// * suspend_resume : After a task submits a request, it suspends itself and is
            /// awoken by the polling thread when the request is tested as ready
            ///
            /// * new_task : after the request is submitted, the task terminates, and the
            /// continuation attached to the completion of the request triggers creation of a
            /// new task which the polling thread inserts into the work queues
            ///
            /// * continuation : the polling thread will call the continuation directly
            ///
            /// * unspecified : reserved for development purposes or for customization by an
            /// application using pika
            yield_while = 0b0000'0000,                                          // 0x00, 00 -> 7
            suspend_resume = 0b0000'1000,                                       // 0x08, 08 -> 15
            new_task = 0b0001'0000,                                             // 0x10, 16 -> 23
            continuation = 0b0001'1000,                                         // 0x18, 24 -> 31
            mpix_continuation = 0b0010'0000,                                    // 0x20, 32 -> 39
            default_mode = continuation + completion_inline + high_priority,    // 24 + 2 + 4 = 30
        };

        /// 3 bits define continuation mode
        inline handler_method get_handler_method(std::underlying_type_t<handler_method> flags)
        {
            return static_cast<handler_method>(
                flags & pika::detail::to_underlying(handler_method::method_mask));
        }

        /// 1 bit defines high priority boost mode for pool transfers
        inline bool use_priority_boost(std::size_t mode)
        {
            return (mode & pika::detail::to_underlying(handler_method::high_priority)) != 0;
        }
        /// 1 bit defines inline or transfer completion
        inline bool use_inline_completion(std::size_t mode)
        {
            return (mode & pika::detail::to_underlying(handler_method::completion_inline)) != 0;
        }
        /// 1 bit defines inline or transfer mpi invocation
        inline bool use_inline_request(std::size_t mode)
        {
            return (mode & pika::detail::to_underlying(handler_method::request_inline)) != 0;
        }

        /// used for debugging to show mode type in messages, should be removed
        inline char const* mode_string(std::size_t flags)
        {
            switch (get_handler_method(flags))
            {
            case handler_method::yield_while: return "yield_while";
            case handler_method::new_task: return "new_task";
            case handler_method::continuation: return "continuation";
            case handler_method::suspend_resume: return "suspend_resume";
            case handler_method::mpix_continuation: return "mpix_continuation";
            default: return "invalid";
            }
        }

        /// mpix extensions in openmpi to support mpi continuations
        using MPIX_Continue_cb_function = int(int rc, void* cb_data);
        PIKA_EXPORT void register_mpix_continuation(
            MPI_Request*, MPIX_Continue_cb_function*, void*);
        /// called after each completed continuation to restart/re-enable continuation support
        PIKA_EXPORT void restart_mpix();

        // -----------------------------------------------------------------
        /// called at runtime start when command-line flags ask for an mpi pool
        void init_resource_partitioner_handler(pika::resource::partitioner&,
            pika::program_options::variables_map const& vm, polling_pool_creation_mode mode);

        // -----------------------------------------------------------------
        /// creates a pool to be used for mpi polling, returns true if the pool was created
        /// and false if it was not, due to lack of threads, or running on a single rank
        /// passing a pool creation mode of force create will only fail if insufficient threads
        /// exist to support one
        PIKA_EXPORT bool create_pool(pika::resource::partitioner& rp, std::string const& pool_name,
            polling_pool_creation_mode mode);

        // -----------------------------------------------------------------
        /// tell the pika::mpi frework which pool is being used for mpi polling
        PIKA_EXPORT void register_pool(std::string const& pool_name);

        // -----------------------------------------------------------------
        /// actually enable/disable the polling callback handler
        PIKA_EXPORT void register_polling();
        PIKA_EXPORT void unregister_polling();

        // -----------------------------------------------------------------
        /// set the maximum number of MPI_Request completions to handle at each polling event
        PIKA_EXPORT void set_max_polling_size(std::size_t);
        PIKA_EXPORT std::size_t get_max_polling_size();

        // -----------------------------------------------------------------
        /// Set/Get the pool_enabled flag
        PIKA_EXPORT bool get_pool_enabled();
        PIKA_EXPORT void set_pool_enabled(bool);

        // -----------------------------------------------------------------
        /// Set/Get the polling and handler mode for continuations.
        PIKA_EXPORT void set_completion_mode(std::size_t);

    }    // namespace detail

    /// when true pika::mpi can disable pool creation, or change the thread mode
    /// otherwise, the flags passed by the user to completion mode etc are honoured
    PIKA_EXPORT void enable_optimizations(bool);

    // -----------------------------------------------------------------
    /// Set/Get the polling and handler mode for continuations.
    PIKA_EXPORT std::size_t get_completion_mode();

    /// can be used to choose between single/multi thread model when initializing MPI
    PIKA_EXPORT int get_preferred_thread_mode();

    /// return the total number of mpi requests currently in queues
    /// This number is an estimate as new work might be created/added during the call
    /// and so the returned value should be considered approximate (racy)
    PIKA_EXPORT size_t get_work_count();

    /// return the name assigned to the mpi polling pool
    PIKA_EXPORT std::string const& get_pool_name();

    /// initialize the pika::mpi background request handler
    /// All ranks should call this function (but only one thread per rank needs to do so)
    PIKA_EXPORT void start_polling(
        exception_mode errorhandler = no_handler, std::string pool_name = "");

    PIKA_EXPORT void stop_polling();

    // -----------------------------------------------------------------
    /// This RAII helper class ensures that MPI polling start/stop is handled correctly
    struct [[nodiscard]] enable_polling
    {
        /// an empty pool name tells pika to use the mpi pool if it exists
        explicit enable_polling(
            exception_mode errorhandler = no_handler, std::string const& pool_name = "")
        {
            start_polling(errorhandler, pool_name);
        }

        ~enable_polling() { stop_polling(); }
    };
}    // namespace pika::mpi::experimental
