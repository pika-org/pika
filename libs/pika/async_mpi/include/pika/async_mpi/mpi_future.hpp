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

namespace pika { namespace mpi { namespace experimental {

    // -----------------------------------------------------------------
    // by convention the title is 7 chars (for alignment)
    using print_on = debug::enable_print<false>;
    static constexpr print_on mpi_debug("MPI_FUT");

    // -----------------------------------------------------------------
    namespace detail {

        using request_callback_function_type =
            pika::util::unique_function<void(int)>;

        PIKA_EXPORT void add_request_callback(
            request_callback_function_type&& f, MPI_Request req);
        PIKA_EXPORT void register_polling(pika::threads::thread_pool_base&);
        PIKA_EXPORT void unregister_polling(pika::threads::thread_pool_base&);

        // -----------------------------------------------------------------
        // An implementation of future_data for MPI
        struct future_data : pika::lcos::detail::future_data<int>
        {
            PIKA_NON_COPYABLE(future_data);

            using init_no_addref =
                typename pika::lcos::detail::future_data<int>::init_no_addref;

            // default empty constructor
            future_data() = default;

            // constructor that takes a request
            future_data(init_no_addref no_addref, MPI_Request request)
              : pika::lcos::detail::future_data<int>(no_addref)
              , request_(request)
            {
                add_callback();
            }

            // constructor used for creation directly by invoke
            future_data(init_no_addref no_addref)
              : pika::lcos::detail::future_data<int>(no_addref)
            {
            }

            // Used when the request was not available when constructing
            // future_data
            void add_callback()
            {
                add_request_callback(
                    [fdp = pika::memory::intrusive_ptr<future_data>(this)](
                        int status) {
                        if (status == MPI_SUCCESS)
                        {
                            // mark the future as ready by setting the shared_state
                            fdp->set_data(MPI_SUCCESS);
                        }
                        else
                        {
                            fdp->set_exception(
                                std::make_exception_ptr(mpi_exception(status)));
                        }
                    },
                    request_);
            }

            // The native MPI request handle owned by this future data
            MPI_Request request_;
        };

        // -----------------------------------------------------------------
        // intrusive pointer for future_data
        using future_data_ptr = memory::intrusive_ptr<future_data>;

        // -----------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors if we enable the error handler
        PIKA_EXPORT extern MPI_Errhandler pika_mpi_errhandler;

        // function that converts an MPI error into an exception
        PIKA_EXPORT void pika_MPI_Handler(MPI_Comm*, int* errorcode, ...);

        // -----------------------------------------------------------------
        /// Called by the mpi senders/executors to initiate throttling
        /// when necessary
        PIKA_EXPORT void wait_for_throttling();

    }    // namespace detail

    // -----------------------------------------------------------------
    /// Set the number of messages above which throttling will be applied
    /// when invoking an MPI function. If the number of messages in flight
    /// exceeds the amount specified, then any thread attempting to invoke
    /// and MPI function that generates an MPI_Request will be suspended.
    /// This should be used with great caution as setting it too low can
    /// cause deadlocks. The default value is size_t(-1) - i.e. unlimited
    /// The value can be set using an environment variable as folows
    /// PIKA_MPI_MSG_THROTTLE=512
    /// but user code setting it will override any default or env value
    /// This function returns the previous throttling threshold value
    PIKA_EXPORT size_t set_max_requests_in_flight(size_t);

    /// Query the current value of the throttling threshold
    PIKA_EXPORT size_t get_max_requests_in_flight();

    // -----------------------------------------------------------------
    /// returns the number of mpi requests currently outstanding
    PIKA_EXPORT size_t get_num_requests_in_flight();

    // -----------------------------------------------------------------
    // set an error handler for communicators that will be called
    // on any error instead of the default behavior of program termination
    PIKA_EXPORT void set_error_handler();

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    PIKA_EXPORT pika::future<void> get_future(MPI_Request request);

    // -----------------------------------------------------------------
    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets ready state in waiting receivers
    PIKA_EXPORT pika::threads::policies::detail::polling_status poll();

    // -----------------------------------------------------------------
    // return a future from an async call to MPI_Ixxx function
    namespace detail {

        template <typename F, typename... Ts>
        pika::future<int> async(F f, Ts&&... ts)
        {
            // create a future data shared state
            detail::future_data_ptr data =
                new detail::future_data(detail::future_data::init_no_addref{});

            // invoke the call to MPI_Ixxx, ignore the returned result for now
            int result = f(PIKA_FORWARD(Ts, ts)..., &data->request_);
            PIKA_UNUSED(result);

            // Add callback after the request has been filled
            data->add_callback();

            // return a future bound to the shared state
            using traits::future_access;
            return future_access<pika::future<int>>::create(PIKA_MOVE(data));
        }
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

    // -----------------------------------------------------------------
    //    template <typename... Args>
    //    inline void debug(const char *title, Args&&... args)
    //    {
    //        if constexpr (mpi_debug.is_enabled())
    //                mpi_debug.debug(debug::str<>(title),
    //                                detail::get_mpi_info(), PIKA_FORWARD(Args, args)...);
    //    }
}}}    // namespace pika::mpi::experimental
