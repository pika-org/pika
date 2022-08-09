//  Copyright (c) 2019 John Biddiscombe
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
#include <pika/modules/threading_base.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/runtime/thread_pool_helpers.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace pika::mpi::experimental {

    enum stream_index : std::uint32_t
    {
        default_stream = 0,
        send_stream,
        receive_stream,
        collective_stream,
        user_stream,
    };

    namespace detail {
        // -----------------------------------------------------------------
        // by convention the title is 7 chars (for alignment)
        using print_on = pika::debug::detail::enable_print<false>;
        static constexpr print_on mpi_debug("MPI_FUT");
        struct mpi_queue;

        using request_callback_function_type =
            pika::util::detail::unique_function<void(int)>;

        PIKA_EXPORT void add_request_callback(
            request_callback_function_type&& f, MPI_Request req,
            stream_index queue);
        PIKA_EXPORT void register_polling(pika::threads::thread_pool_base&);
        PIKA_EXPORT void unregister_polling(pika::threads::thread_pool_base&);

        // -----------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors if we enable the error handler
        PIKA_EXPORT extern MPI_Errhandler pika_mpi_errhandler;

        // function that converts an MPI error into an exception
        PIKA_EXPORT void pika_MPI_Handler(MPI_Comm*, int* errorcode, ...);

        // -----------------------------------------------------------------
        /// Called by the mpi senders/executors to initiate throttling
        /// when necessary
        PIKA_EXPORT void wait_for_throttling_snd(mpi_queue& queue);
        PIKA_EXPORT void wait_for_throttling_snd(stream_index stream);

        // -----------------------------------------------------------------
        /// Set the number of messages above which throttling will be applied
        /// when invoking an MPI function. If the number of messages in flight
        /// exceeds the amount specified, then any thread attempting to invoke
        /// and MPI function that generates an MPI_Request will be suspended.
        /// This should be used with great caution as setting it too low can
        /// cause deadlocks. The default value is size_t(-1) - i.e. unlimited
        /// The value can be set using an environment variable as follows
        /// PIKA_MPI_MSG_THROTTLE=64
        /// but user code setting it will override any default or env value
        /// This function returns the previous throttling threshold value
        PIKA_EXPORT std::uint32_t set_max_requests_in_flight(
            std::uint32_t, stream_index = stream_index(-1));

        /// Query the current value of the throttling threshold
        PIKA_EXPORT std::uint32_t get_max_requests_in_flight(stream_index);

        // -----------------------------------------------------------------
        /// returns the number of mpi requests currently outstanding
        PIKA_EXPORT std::uint32_t get_num_requests_in_flight();

        // -----------------------------------------------------------------
        // set an error handler for communicators that will be called
        // on any error instead of the default behavior of program termination
        PIKA_EXPORT void set_error_handler();

        // -----------------------------------------------------------------
        // Background progress function for MPI async operations
        // Checks for completed MPI_Requests and sets ready state in waiting receivers
        PIKA_EXPORT pika::threads::policies::detail::polling_status poll();
    }    // namespace detail

    // initialize the pika::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    PIKA_EXPORT void init(bool init_mpi = false,
        std::string const& pool_name = "", bool init_errorhandler = false);

    // -----------------------------------------------------------------
    PIKA_EXPORT void finalize(std::string const& pool_name = "");

    // -----------------------------------------------------------------
    // This RAII helper class assumes that MPI initialization/finalization is
    // handled elsewhere
    struct [[nodiscard]] enable_user_polling
    {
        enable_user_polling(
            std::string const& pool_name = "", bool init_errorhandler = false)
          : pool_name_(pool_name)
        {
            mpi::experimental::init(false, pool_name, init_errorhandler);
        }

        ~enable_user_polling()
        {
            mpi::experimental::finalize(pool_name_);
        }

    private:
        std::string pool_name_;
    };
}    // namespace pika::mpi::experimental
