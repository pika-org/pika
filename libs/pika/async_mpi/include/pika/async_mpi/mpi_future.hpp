//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_mpi/mpi_exception.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/futures/future.hpp>
#include <pika/modules/concurrency.hpp>
#include <pika/modules/execution_base.hpp>
#include <pika/modules/memory.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/mpi_base/mpi.hpp>
#include <pika/runtime_local/thread_pool_helpers.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace pika { namespace mpi { namespace experimental {

    // -----------------------------------------------------------------
    namespace detail {

        using request_callback_function_type =
            pika::util::unique_function_nonser<void(int)>;

        PIKA_EXPORT void add_request_callback(
            request_callback_function_type&& f, MPI_Request req);
        PIKA_EXPORT void register_polling(pika::threads::thread_pool_base&);
        PIKA_EXPORT void unregister_polling(
            pika::threads::thread_pool_base&);

    }    // namespace detail

    // by convention the title is 7 chars (for alignment)
    using print_on = debug::enable_print<false>;
    static constexpr print_on mpi_debug("MPI_FUT");

    namespace detail {

        using mutex_type = pika::lcos::local::spinlock;

        // mutex needed to protect mpi request vector, note that the
        // mpi poll function takes place inside the main scheduling loop
        // of pika and not on an pika worker thread, so we must use std:mutex
        PIKA_EXPORT mutex_type& get_vector_mtx();

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
        // a convenience structure to hold state vars
        // used extensivey with debug::print to display rank etc
        struct mpi_info
        {
            bool error_handler_initialized_ = false;
            int rank_ = -1;
            int size_ = -1;
            // requests vector holds the requests that are checked
            std::atomic<std::uint32_t> requests_vector_size_{0};
            // requests queue holds the requests recently added
            std::atomic<std::uint32_t> requests_queue_size_{0};
        };

        // an instance of mpi_info that we store data in
        PIKA_EXPORT mpi_info& get_mpi_info();

        // stream operator to display debug mpi_info
        PIKA_EXPORT std::ostream& operator<<(
            std::ostream& os, mpi_info const& i);

        // -----------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors is we enable the error handler
        PIKA_EXPORT extern MPI_Errhandler pika_mpi_errhandler;

        // function that converts an MPI error into an exception
        PIKA_EXPORT void pika_MPI_Handler(MPI_Comm*, int* errorcode, ...);

        // -----------------------------------------------------------------
        // we track requests and callbacks in two vectors even though
        // we have the request stored in the request_callback vector already
        // the reason for this is because we can use MPI_Testany
        // with a vector of requests to save overheads compared
        // to testing one by one every item (using a list)
        PIKA_EXPORT std::vector<MPI_Request>& get_requests_vector();

        // -----------------------------------------------------------------
        // define a lockfree queue type to place requests in prior to handling
        // this is done only to avoid taking a lock every time a request is
        // returned from MPI. Instead the requests are placed into a queue
        // and the polling code pops them prior to calling Testany
        using queue_type = concurrency::ConcurrentQueue<future_data_ptr>;

        // -----------------------------------------------------------------
        // used internally to query how many requests are 'in flight'
        // these are requests that are being polled for actively
        // and not the same as the requests enqueued
        PIKA_EXPORT std::size_t get_number_of_active_requests();

    }    // namespace detail

    // -----------------------------------------------------------------
    // set an error handler for communicators that will be called
    // on any error instead of the default behavior of program termination
    PIKA_EXPORT void set_error_handler();

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    PIKA_EXPORT pika::future<void> get_future(MPI_Request request);

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

    // -----------------------------------------------------------------
    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets mpi::experimental::future ready
    // when found
    PIKA_EXPORT pika::threads::policies::detail::polling_status poll();

    // -----------------------------------------------------------------
    // This is not completely safe as it will return when the request vector is
    // empty, but cannot guarantee that other communications are not about
    // to be launched in outstanding continuations etc.
    inline void wait()
    {
        pika::util::yield_while([]() {
            std::unique_lock<detail::mutex_type> lk(
                detail::get_vector_mtx(), std::try_to_lock);
            if (!lk.owns_lock())
            {
                return true;
            }
            return (detail::get_mpi_info().requests_vector_size_ > 0);
        });
    }

    template <typename F>
    inline void wait(F&& f)
    {
        pika::util::yield_while([&]() {
            std::unique_lock<detail::mutex_type> lk(
                detail::get_vector_mtx(), std::try_to_lock);
            if (!lk.owns_lock())
            {
                return true;
            }
            return (detail::get_mpi_info().requests_vector_size_ > 0) || f();
        });
    }

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
    struct PIKA_NODISCARD enable_user_polling
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
    template <typename... Args>
    inline void debug(Args&&... args)
    {
        mpi_debug.debug(detail::get_mpi_info(), PIKA_FORWARD(Args, args)...);
    }
}}}    // namespace pika::mpi::experimental
