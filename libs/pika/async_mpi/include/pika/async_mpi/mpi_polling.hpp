//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_mpi/mpi_exception.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/futures/future.hpp>
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

namespace pika::mpi::experimental {

    template <typename E>
    constexpr std::underlying_type_t<E> to_underlying(E e) noexcept
    {
        return static_cast<std::underlying_type_t<E>>(e);
    }

    enum pool_create_mode
    {
        pika_decides = 0,
        force_create,
        force_no_create,
    };

    enum class stream_type : std::uint32_t
    {
        automatic = 0,
        send_1,
        send_2,
        receive_1,
        receive_2,
        collective_1,
        collective_2,
        user_1,
        user_2,
        max_stream
    };

    namespace detail {
        using request_callback_function_type = pika::util::detail::unique_function<void(int)>;

        const char* stream_name(stream_type s);

        PIKA_EXPORT bool add_request_callback(request_callback_function_type&&, MPI_Request);
        PIKA_EXPORT void register_polling(pika::threads::detail::thread_pool_base&);
        PIKA_EXPORT void unregister_polling(pika::threads::detail::thread_pool_base&);

        PIKA_EXPORT std::size_t get_completion_mode_default();

        // -----------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors if we enable the error handler
        PIKA_EXPORT extern MPI_Errhandler pika_mpi_errhandler;

        // function that converts an MPI error into an exception
        PIKA_EXPORT void pika_MPI_Handler(MPI_Comm*, int* errorcode, ...);

        // -----------------------------------------------------------------
        // set an error handler for communicators that will be called
        // on any error instead of the default behavior of program termination
        PIKA_EXPORT void set_error_handler();

        // -----------------------------------------------------------------
        // Background progress function for MPI async operations
        // Checks for completed MPI_Requests and sets ready state in waiting receivers
        PIKA_EXPORT pika::threads::detail::polling_status poll();

        // utility function to avoid duplication in eager check locations
        PIKA_EXPORT bool poll_request(MPI_Request /*req*/);

        // -----------------------------------------------------------------
        using semaphore_type = pika::counting_semaphore<>;
        //
        PIKA_EXPORT std::shared_ptr<semaphore_type> get_semaphore(stream_type s);

        inline constexpr bool throttling_enabled = false;

        /// flags that control how mpi continuations are handled
        enum class handler_mode : std::uint32_t
        {
            /// enable the use of a dedicated pool for polling mpi messages.
            use_pool = 0x01,

            /// this bit enables the inline invocation of the mpi request, when set
            /// the calling thread performs the mpi operation, when unset, a transfer
            /// is made so that the invocation happens on a new task that
            /// would normally be on a dedicated pool if/when it exists
            request_inline = 0x02,

            /// this bit enables the inline execution of the completion handler for the
            /// request, when unset a transfer is made to move the completion handler
            /// from the polling thread onto a new one
            completion_inline = 0x04,

            /// this bit enables the use of a high priority task flag when a
            /// completion is handled so that the continuation is executed quickly
            /// if transferred to a new task, or resumed from sleep.
            high_priority = 0x08,

            /// 3 bits control the handler method,
            method_mask = 0x70,

            /// the individual methods that are supported for dispatching continuations
            yield_while = 0x00,
            suspend_resume = 0x10,
            new_task = 0x20,
            continuation = 0x30,
            original = 0x40,

            /// Default flags are to invoke inline, but transfer completion using a dedicated pool
            default_mode = use_pool | request_inline | high_priority | new_task,
        };

        // 2 bits define continuation mode
        inline handler_mode get_handler_mode(int flags)
        {
            return static_cast<handler_mode>(flags & to_underlying(handler_mode::method_mask));
        }

        // 1 bit defines high priority mode for completion
        inline bool use_HP_completion(int mode)
        {
            return static_cast<bool>((mode & to_underlying(handler_mode::high_priority)) ==
                to_underlying(handler_mode::high_priority));
        }
        // 1 bit defines inline or transfer completion
        inline bool use_inline_completion(int mode)
        {
            return static_cast<bool>((mode & to_underlying(handler_mode::completion_inline)) ==
                to_underlying(handler_mode::completion_inline));
        }
        // 1 bit defines inline or transfer mpi invocation
        inline bool use_inline_request(int mode)
        {
            return static_cast<bool>((mode & to_underlying(handler_mode::request_inline)) ==
                to_underlying(handler_mode::request_inline));
        }
        // 1 bit defines whether we use a pool or not
        inline bool use_pool(int mode)
        {
            return static_cast<bool>((mode & to_underlying(handler_mode::use_pool)) ==
                to_underlying(handler_mode::use_pool));
        }

        inline const char* mode_string(int flags)
        {
            switch (get_handler_mode(flags))
            {
            case handler_mode::yield_while:
                return "yield_while";
                break;
            case handler_mode::new_task:
                return "new_task";
                break;
            case handler_mode::continuation:
                return "continuation";
                break;
            case handler_mode::suspend_resume:
                return "suspend_resume";
                break;
            default:
                return "default";
            }
        }

        // needed by static checks when debugging
        PIKA_EXPORT int comm_world_size();

    }    // namespace detail

    // -----------------------------------------------------------------
    /// Set the number of messages above which throttling will be applied
    /// when invoking an MPI function on a given stream.
    /// If the number of messages in flight exceeds the amount specified,
    /// then any thread attempting to invoke an MPI function on that stream
    /// that generates an MPI_Request will be suspended.
    /// This should be used with great caution as setting it too low can
    /// cause deadlocks. The default value is size_t(-1) - i.e. unlimited
    /// The value can be set using an environment variable as follows
    /// PIKA_MPI_MSG_THROTTLE=64
    /// but user code setting it will override any default or env value
    /// If the optional stream param is not specified, then all streams
    /// are set with the same limit
    PIKA_EXPORT void set_max_requests_in_flight(
        std::uint32_t, std::optional<stream_type> = std::nullopt);

    /// Query the current value of the throttling threshold for the stream
    /// if the optional stream param is not specified, then the
    /// automatic/default stream value is returned
    PIKA_EXPORT std::uint32_t get_max_requests_in_flight(std::optional<stream_type> = std::nullopt);

    /// return the total number of mpi requests currently in queues
    /// This number is an estimate as new work might be created/added during the call
    /// and so the returned value should be considered approximate (racy)
    PIKA_EXPORT size_t get_work_count();

    // -----------------------------------------------------------------
    /// set the maximume number of MPI_Request completions to
    /// handle at each polling event
    PIKA_EXPORT void set_max_polling_size(std::size_t);
    PIKA_EXPORT std::size_t get_max_polling_size();

    // -----------------------------------------------------------------
    /// Get the poll transfer mode. when an mpi message completes,
    /// it may trigger a continuation,
    /// mode 0 - the continuation is run inline directly on the polling
    /// thread of the pool doing the polling
    /// mode 1 - the continuation is wrapped into a high priority task
    /// and placed in the queue on the default pool
    /// mode 2 - the continuation is wrapped into a high priority task
    /// and placed in the queue on the pool doing the polling (same as mode 1
    /// if no mpi pool enabled)
    /// mode 3 - the continuation is wrapped into a task but run inline on
    /// whichever pool is doing the polling, bypassing queues altogether
    PIKA_EXPORT std::size_t get_completion_mode();

    PIKA_EXPORT bool create_pool(pika::resource::partitioner&, std::string const& = "",
        pool_create_mode = pool_create_mode::pika_decides);

    PIKA_EXPORT const std::string& get_pool_name();
    PIKA_EXPORT void set_pool_name(const std::string&);

    // returns false if no custom mpi pool has been created
    PIKA_EXPORT bool pool_exists();

    // initialize the pika::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    PIKA_EXPORT void init(bool init_mpi = false, bool init_errorhandler = false);

    // -----------------------------------------------------------------
    PIKA_EXPORT void finalize(std::string const& pool_name = "");

    // -----------------------------------------------------------------
    // This RAII helper class assumes that MPI initialization/finalization is
    // handled elsewhere
    struct [[nodiscard]] enable_user_polling
    {
        enable_user_polling(std::string const& pool_name = "", bool init_errorhandler = false)
          : pool_name_(pool_name)
        {
            mpi::experimental::init(false, init_errorhandler);
        }

        ~enable_user_polling()
        {
            mpi::experimental::finalize(pool_name_);
        }

    private:
        std::string pool_name_;
    };
}    // namespace pika::mpi::experimental
