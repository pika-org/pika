//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_mpi/mpi_exception.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/modules/concurrency.hpp>
#include <pika/modules/execution_base.hpp>
#include <pika/modules/memory.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/mpi_base/mpi.hpp>
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

    enum pool_create_mode
    {
        pika_decides = 0,
        force_create,
        force_no_create,
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

        // -----------------------------------------------------------------
        /// Background progress function for MPI async operations
        /// Checks for completed MPI_Requests and sets ready state in waiting receivers
        PIKA_EXPORT pika::threads::detail::polling_status poll();

        /// utility function to avoid duplication in eager check locations
        PIKA_EXPORT bool poll_request(MPI_Request /*req*/);

        // -----------------------------------------------------------------
        // handling of requests and continuations
        /// flags that control how mpi continuations are handled
        enum class handler_method : std::uint32_t
        {
            /// enable the use of a dedicated pool for polling mpi messages.
            use_pool = 0b0000'0001,    // 01

            /// this bit enables the inline invocation of the mpi request, when set
            /// the calling thread performs the mpi operation, when unset, a transfer
            /// is made so that the invocation happens on a new task that
            /// would normally be on a dedicated pool if/when it exists
            request_inline = 0b0000'0010,    // 02

            /// this bit enables the inline execution of the completion handler for the
            /// request, when unset a transfer is made to move the completion handler
            /// from the polling thread onto a new one
            completion_inline = 0b0000'0100,    // 04

            /// this bit enables the use of a high priority task flag
            /// 1) requests are boosted to high priority if they are passed the the mpi-pool
            ///    to ensure they execute before other polling tasks (reduce latency)
            /// 2) completions are boosted to high priority when sent to the main thread pool
            ///    so that the continuation is executed as quickly as possible
            high_priority = 0b0000'1000,    // 08

            /// 3 bits control the handler method,
            method_mask = 0b0111'0000,    // 70

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
            yield_while = 0b0000'0000,               // 0x00, 00 ... 15
            suspend_resume = 0b0001'0000,            // 0x10, 16 ... 31
            new_task = 0b0010'0000,                  // 0x20, 32 ... 47
            continuation = 0b0011'0000,              // 0x30, 48 ... 63
            mpix_continuation = 0b0100'0000,         // 0x40, 64 ... 79
            unspecified = mpix_continuation + 16,    // 0x50, ...

            /// Default flags are to invoke inline, but transfer completion using a dedicated pool
            default_mode = use_pool | request_inline | high_priority | new_task,
        };

        /// 3 bits define continuation mode
        inline handler_method get_handler_method(const std::underlying_type_t<handler_method> flags)
        {
            return static_cast<handler_method>(
                flags & pika::detail::to_underlying(handler_method::method_mask));
        }

        /// 1 bit defines high priority boost mode for pool transfers
        inline bool use_priority_boost(const int mode)
        {
            return (mode & pika::detail::to_underlying(handler_method::high_priority)) != 0;
        }
        /// 1 bit defines inline or transfer completion
        inline bool use_inline_completion(const int mode)
        {
            return (mode & pika::detail::to_underlying(handler_method::completion_inline)) != 0;
        }
        /// 1 bit defines inline or transfer mpi invocation
        inline bool use_inline_request(const int mode)
        {
            return (mode & pika::detail::to_underlying(handler_method::request_inline)) != 0;
        }
        /// 1 bit defines whether we use a pool or not
        inline bool use_pool(const int mode)
        {
            return (mode & pika::detail::to_underlying(handler_method::use_pool)) != 0;
        }
        /// Convenience fn to test if mode supports inline continuation
        inline bool inline_ready(const int mode)
        {
            handler_method h = get_handler_method(mode);
            /// these task modes always trigger continuations on a pika task and can be safely inlined
            return (h == handler_method::yield_while) ||
                (h == handler_method::suspend_resume) | (h == handler_method::new_task);
        }

        /// used for debugging to show mode type in messages, should be removed
        inline const char* mode_string(int flags)
        {
            switch (get_handler_method(flags))
            {
            case handler_method::yield_while: return "yield_while";
            case handler_method::new_task: return "new_task";
            case handler_method::continuation: return "continuation";
            case handler_method::suspend_resume: return "suspend_resume";
            case handler_method::mpix_continuation: return "mpix_continuation";
            case handler_method::unspecified: return "unspecified";
            default: return "invalid";
            }
        }

        /// utility : needed by static checks when debugging
        PIKA_EXPORT int comm_world_size();

        /// mpix extensions in openmpi to support mpi continuations
        using MPIX_Continue_cb_function = int(int rc, void* cb_data);
        PIKA_EXPORT void register_mpix_continuation(
            MPI_Request*, MPIX_Continue_cb_function*, void*);
        /// called after each completed continuation to restart/re-enable continuation support
        PIKA_EXPORT void restart_mpix();
    }    // namespace detail

    /// return the total number of mpi requests currently in queues
    /// This number is an estimate as new work might be created/added during the call
    /// and so the returned value should be considered approximate (racy)
    PIKA_EXPORT size_t get_work_count();

    // -----------------------------------------------------------------
    /// set the maximum number of MPI_Request completions to handle at each polling event
    PIKA_EXPORT void set_max_polling_size(std::size_t);
    PIKA_EXPORT std::size_t get_max_polling_size();

    // -----------------------------------------------------------------
    /// Get the polling transfer mode for continuations.
    PIKA_EXPORT std::size_t get_completion_mode();
    PIKA_EXPORT void set_completion_mode(std::size_t mode);

    PIKA_EXPORT bool create_pool(
        std::string const& = "", pool_create_mode = pool_create_mode::pika_decides);

    PIKA_EXPORT const std::string& get_pool_name();
    PIKA_EXPORT void set_pool_name(const std::string&);

    // returns false if no custom mpi pool has been created
    PIKA_EXPORT bool pool_exists();

    PIKA_EXPORT void register_polling();

    PIKA_EXPORT int get_preferred_thread_mode();

    /// when true pika::mpi can disable the pool, or change the thread mode
    /// otherwise, the flags passed by the user to completion mode etc are honoured
    PIKA_EXPORT void enable_optimizations(bool enable);

    // initialize the pika::mpi background request handler
    // All ranks should call this function (but only one thread per rank needs to do so)
    PIKA_EXPORT void init(bool init_mpi = false, bool init_errorhandler = false);

    // -----------------------------------------------------------------
    PIKA_EXPORT void finalize(std::string const& pool_name = "");

    // -----------------------------------------------------------------
    // This RAII helper class assumes that MPI initialization/finalization is handled elsewhere
    struct [[nodiscard]] enable_user_polling
    {
        enable_user_polling(std::string const& pool_name = "", bool init_errorhandler = false)
          : pool_name_(pool_name)
        {
            mpi::experimental::init(false, init_errorhandler);
            mpi::experimental::register_polling();
        }

        ~enable_user_polling() { mpi::experimental::finalize(pool_name_); }

    private:
        std::string pool_name_;
    };
}    // namespace pika::mpi::experimental
