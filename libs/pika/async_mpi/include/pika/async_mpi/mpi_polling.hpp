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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pika::mpi::experimental::detail {
    template <typename E>
    constexpr std::underlying_type_t<E> to_underlying(E e) noexcept
    {
        return static_cast<std::underlying_type_t<E>>(e);
    }
}    // namespace pika::mpi::experimental::detail

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
        enum class handler_mode : std::uint32_t
        {
            /// enable the use of a dedicated pool for polling mpi messages.
            use_pool = 0b0000'0001,    // 1

            /// this bit enables the inline invocation of the mpi request, when set
            /// the calling thread performs the mpi operation, when unset, a transfer
            /// is made so that the invocation happens on a new task that
            /// would normally be on a dedicated pool if/when it exists
            request_inline = 0b0000'0010,    // 2

            /// this bit enables the inline execution of the completion handler for the
            /// request, when unset a transfer is made to move the completion handler
            /// from the polling thread onto a new one
            completion_inline = 0b0000'0100,    // 4

            /// this bit enables the use of a high priority task flag
            /// 1) requests are boosted to high priority if they are passed the the mpi-pool
            ///    to ensure they execute before other polling tasks (reduce latency)
            /// 2) completions are boosted to high priority when sent to the main thread pool
            ///    so that the continuation is executed as quickly as possible
            high_priority = 0b0000'1000,    // 8

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
            yield_while = 0b0000'0000,       // 0x00, 00 ... 15
            suspend_resume = 0b0001'0000,    // 0x10, 16 ... 31
            new_task = 0b0010'0000,          // 0x20, 32 ... 47
            continuation = 0b0011'0000,      // 0x30, 48 ... 63
            unspecified = 0b0100'0000,       // 0x40, 64 ...

            /// Default flags are to invoke inline, but transfer completion using a dedicated pool
            default_mode = use_pool | request_inline | high_priority | new_task,
        };

        /// 2 bits define continuation mode
        inline handler_mode get_handler_mode(std::underlying_type_t<handler_mode> flags)
        {
            return static_cast<handler_mode>(
                flags & detail::to_underlying(handler_mode::method_mask));
        }

        /// 1 bit defines high priority boost mode for pool transfers
        inline bool use_priority_boost(int mode)
        {
            return static_cast<bool>((mode & detail::to_underlying(handler_mode::high_priority)) ==
                detail::to_underlying(handler_mode::high_priority));
        }
        /// 1 bit defines inline or transfer completion
        inline bool use_inline_completion(int mode)
        {
            return static_cast<bool>(
                (mode & detail::to_underlying(handler_mode::completion_inline)) ==
                detail::to_underlying(handler_mode::completion_inline));
        }
        /// 1 bit defines inline or transfer mpi invocation
        inline bool use_inline_request(int mode)
        {
            return static_cast<bool>((mode & detail::to_underlying(handler_mode::request_inline)) ==
                detail::to_underlying(handler_mode::request_inline));
        }
        /// 1 bit defines whether we use a pool or not
        inline bool use_pool(int mode)
        {
            return static_cast<bool>((mode & detail::to_underlying(handler_mode::use_pool)) ==
                detail::to_underlying(handler_mode::use_pool));
        }

        /// used for debugging to show mode type in messages, should be removed
        inline const char* mode_string(int flags)
        {
            switch (get_handler_mode(flags))
            {
            case handler_mode::yield_while: return "yield_while"; break;
            case handler_mode::new_task: return "new_task"; break;
            case handler_mode::continuation: return "continuation"; break;
            case handler_mode::suspend_resume: return "suspend_resume"; break;
            case handler_mode::unspecified: return "unspecified"; break;
            default: return "invalid";
            }
        }

        /// utility : needed by static checks when debugging
        PIKA_EXPORT int comm_world_size();

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

    PIKA_EXPORT bool create_pool(
        std::string const& = "", pool_create_mode = pool_create_mode::pika_decides);

    PIKA_EXPORT const std::string& get_pool_name();
    PIKA_EXPORT void set_pool_name(const std::string&);

    // returns false if no custom mpi pool has been created
    PIKA_EXPORT bool pool_exists();

    PIKA_EXPORT void register_polling();

    // initialize the pika::mpi background request handler
    // All ranks should call this function (but only one thread per rank needs to do so)
    PIKA_EXPORT void init(bool init_mpi = false, bool init_errorhandler = false,
        pool_create_mode pool_mode = pool_create_mode::pika_decides);

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
        }

        ~enable_user_polling() { mpi::experimental::finalize(pool_name_); }

    private:
        std::string pool_name_;
    };
}    // namespace pika::mpi::experimental
