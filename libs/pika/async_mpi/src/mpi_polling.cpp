//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/mpi_exception.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/command_line_handling/get_env_var_as.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/debugging/print.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/mpi_base/mpi_environment.hpp>
#include <pika/resource_partitioner/detail/partitioner.hpp>
#include <pika/string_util/case_conv.hpp>
#include <pika/synchronization/condition_variable.hpp>
#include <pika/synchronization/mutex.hpp>
#include <pika/threading_base/detail/global_activity_count.hpp>
#include <pika/type_support/to_underlying.hpp>
//
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if __has_include(<mpi-ext.h>)
# include <mpi-ext.h>
#endif

#ifdef OMPI_HAVE_MPI_EXT_CONTINUE
namespace pika::mpi::experimental::detail {
    static inline void mpi_ext_continuation_result_check(int code)
    {
        if (code != MPI_SUCCESS)
            throw pika::mpi::experimental::mpi_exception(code, "MPI_EXT_CONTINUATION problem");
    }
}    // namespace pika::mpi::experimental::detail
#endif

namespace pika::mpi::experimental {
    namespace detail {
        // -----------------------------------------------------------------
        // by convention the title is 7 chars (for alignment)
        // a debug level of N shows messages with level 1..N
        template <int Level>
        static debug::detail::print_threshold<Level, 0> mpi_debug("MPIPOLL");

        constexpr std::uint32_t max_poll_requests = 32;

        // -----------------------------------------------------------------
        /// Get the default value for number of requests to poll for in calls to MPI_Test etc
        std::size_t get_polling_default();
        /// Get the default mode for completions/transfers of MPI requests to from pools
        std::size_t get_completion_mode_default();

        // -----------------------------------------------------------------
        /// Holds an MPI_Request and a callback. The callback is intended to be
        /// called when the operation tied to the request handle completes.
        struct request_callback
        {
            MPI_Request request_;
            request_callback_function_type callback_function_;
        };

        // -----------------------------------------------------------------
        struct mpi_callback_info
        {
            request_callback_function_type cb_;
            std::int32_t err_;
            MPI_Request request_;
        };

        struct ready_callback
        {
            request_callback_function_type cb_;
            MPI_Request request_;
            std::int32_t err_;
        };

        // -----------------------------------------------------------------
        /// When a user first initiates an MPI call, a request is generated
        /// and a callback associated with it. We place these on a (lock-free) queue
        /// to avoid taking a lock on every invocation of an MPI function.
        /// When a thread polls for MPI completions, it moves the request_callback(s)
        /// into a vector that is passed to the mpi test function
        using request_callback_queue_type = concurrency::detail::ConcurrentQueue<request_callback>;
        //
        using request_ready_queue_type = concurrency::detail::ConcurrentQueue<ready_callback>;

        // -----------------------------------------------------------------
        /// Spinlock is used as it can be called by OS threads or pika tasks
        using mutex_type = pika::detail::spinlock;

        // -----------------------------------------------------------------
        /// a convenience structure to hold state vars in one place
        struct mpi_data
        {
            bool error_handler_initialized_ = false;
            int rank_ = -1;
            int size_ = -1;
            std::atomic<std::size_t> max_polling_requests{get_polling_default()};

            // The sum of messages in queue + vector
            std::atomic<std::uint32_t> all_in_flight_{0};
            // for debugging of code creating/destroying polling handlers
            std::atomic<std::uint32_t> register_polling_count_{0};

            // Principal storage of requests for polling
            // we track requests and callbacks in two vectors
            // because we can use MPI_Testany/MPI_Testsome
            // with a vector of requests to save overheads compared
            // to testing one by one every item (using a list)
            request_callback_queue_type request_callback_queue_;
            request_ready_queue_type ready_requests_;
            //
            std::vector<MPI_Request> requests_;
            std::vector<mpi_callback_info> callbacks_;

            // mutex needed to protect mpi request vector, note that the
            // mpi poll function usually takes place inside the main scheduling loop
            // though poll may also be called directly by a user task.
            // we use a spinlock for both cases
            mutex_type polling_vector_mtx_;

            // MPI continuations support (Experimental mpi extension)
            MPI_Request mpix_continuations_request{MPI_REQUEST_NULL};
            std::mutex mpix_lock;
            std::atomic<bool> mpix_ready_{false};

            // optimizations allow/disallow
            bool optimizations_{false};
        };

        /// a single instance of all the mpi variables initialized once at startup
        static mpi_data mpi_data_;

        // default completion/handler mode for mpi continuations
        static std::size_t completion_flags_{get_completion_mode_default()};

        static std::string polling_pool_name_{"polling"};
        static bool pool_exists_{false};

        // stream operator to display debug mpi_data
        PIKA_EXPORT std::ostream& operator<<(std::ostream& os, mpi_data const& info)
        {
            using namespace pika::debug::detail;
            // clang-format off
            os << "R "
               << dec<3>(info.rank_) << "/"
               << dec<3>(info.size_)
               << " in_flight " << dec<4>(info.all_in_flight_)
               << " vec_cb "    << dec<4>(info.callbacks_.size())
               << " vec_rq "    << dec<4>(info.requests_.size());
            // clang-format on
            return os;
        }

        // stream operator to display debug mpi_request
        PIKA_EXPORT std::ostream& operator<<(std::ostream& os, MPI_Request const& req)
        {
            // clang-format off
            os <<  "req " << debug::detail::hex<8>(req);
            // clang-format on
            return os;
        }

        // -----------------------------------------------------------------
        // When debugging, it might be useful to know how many
        // MPI_REQUEST_NULL messages are currently in our vector
        inline size_t get_num_null_requests_in_vector()
        {
            std::vector<MPI_Request>& vec = mpi_data_.requests_;
            return std::count_if(
                vec.begin(), vec.end(), [](MPI_Request r) { return r == MPI_REQUEST_NULL; });
        }

        // -----------------------------------------------------------------
        std::size_t get_polling_default()
        {
            std::uint32_t val =
                pika::detail::get_env_var_as<std::uint32_t>("PIKA_MPI_POLLING_SIZE", 8);
            PIKA_DETAIL_DP(mpi_debug<2>, debug(str<>("Poll size"), dec<3>(val)));
            mpi_data_.max_polling_requests = val;
            return val;
        }

        // -----------------------------------------------------------------
        std::size_t get_completion_mode_default()
        {
            // inline continuations are default
            return pika::detail::get_env_var_as<std::size_t>("PIKA_MPI_COMPLETION_MODE", 1);
        }

        // -----------------------------------------------------------------
        /// used internally to add an MPI_Request to the lockfree queue
        /// that will be used by the polling routines to check when requests
        /// have completed
        void add_to_request_callback_queue(request_callback&& req_callback)
        {
            pika::threads::detail::increment_global_activity_count();
            ++mpi_data_.all_in_flight_;
            //
            mpi_data_.request_callback_queue_.enqueue(PIKA_MOVE(req_callback));
            PIKA_DETAIL_DP(
                mpi_debug<5>, debug(str<>("CB queued"), ptr(req_callback.request_), mpi_data_));
        }

        // -----------------------------------------------------------------
        /// used internally to add a request to the main polling vector
        /// that is passed to MPI_Testany. This is only called inside the
        /// polling function when a lock is held, so only one thread
        /// at a time ever enters here
        inline void add_to_request_callback_vector(request_callback&& req_callback)
        {
            mpi_data_.requests_.push_back(req_callback.request_);
            mpi_data_.callbacks_.push_back(
                {PIKA_MOVE(req_callback.callback_function_), MPI_SUCCESS, req_callback.request_});

            // clang-format off
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("CB queue => vector"),
                mpi_data_, ptr(req_callback.request_),
                "nulls", dec<3>(get_num_null_requests_in_vector())
                ));
            // clang-format on
        }

// -------------------------------------------------------------
#if defined(PIKA_DEBUG)
        std::atomic<std::uint32_t>& get_register_polling_count()
        {
            return mpi_data_.register_polling_count_;
        }
#endif

        // -------------------------------------------------------------
        bool add_request_callback(request_callback_function_type&& callback, MPI_Request request)
        {
            PIKA_ASSERT_MSG(get_register_polling_count() != 0,
                "MPI event polling has not been enabled on any pool. Make sure that MPI event "
                "polling is enabled on at least one thread pool.");
            add_to_request_callback_queue(request_callback{request, PIKA_MOVE(callback)});
            return true;
        }

        // -------------------------------------------------------------
        // an MPI error handling type that we can use to intercept
        // MPI errors if we enable the error handler
        MPI_Errhandler pika_mpi_errhandler = 0;

        // -------------------------------------------------------------
        // function that converts an MPI error into an exception
        void pika_MPI_Handler(MPI_Comm*, int* errorcode, ...)
        {
            PIKA_DETAIL_DP(
                mpi_debug<5>, debug(str<>("pika_MPI_Handler"), error_message(*errorcode)));
            throw mpi_exception(*errorcode, error_message(*errorcode));
        }

        // -------------------------------------------------------------
        // set an error handler for communicators that will be called
        // on any error instead of the default behavior of program termination
        void set_error_handler()
        {
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("set_error_handler")));

            MPI_Comm_create_errhandler(detail::pika_MPI_Handler, &detail::pika_mpi_errhandler);
            MPI_Comm_set_errhandler(MPI_COMM_WORLD, detail::pika_mpi_errhandler);
        }

        bool poll_request(MPI_Request req)
        {
            int flag;
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            if (flag) { PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("poll MPI_Test ok"), req)); }
            return flag;
        }

        // -------------------------------------------------------------
        /// Remove all entries in request and callback vectors that are invalid
        /// Ideally, would use a zip iterator to do both using remove_if
        void compact_vectors()
        {
            using detail::mpi_data_;

            size_t const size = mpi_data_.requests_.size();
            size_t pos = size;
            // find index of first NULL request
            for (size_t i = 0; i < size; ++i)
            {
                if (mpi_data_.requests_[i] == MPI_REQUEST_NULL)
                {
                    pos = i;
                    break;
                }
            }
            // move all non NULL requests/callbacks towards beginning of vector.
            for (size_t i = pos + 1; i < size; ++i)
            {
                if (mpi_data_.requests_[i] != MPI_REQUEST_NULL)
                {
                    mpi_data_.requests_[pos] = mpi_data_.requests_[i];
                    mpi_data_.callbacks_[pos] = PIKA_MOVE(mpi_data_.callbacks_[i]);
                    pos++;
                }
            }
            // and trim off the space we didn't need
            mpi_data_.requests_.resize(pos);
            mpi_data_.callbacks_.resize(pos);
        }

#ifdef OMPI_HAVE_MPI_EXT_CONTINUE
        pika::threads::detail::polling_status try_mpix_polling(bool singlethreaded)
        {
            PIKA_DETAIL_DP(mpi_debug<5>,
                debug(str<>("mpix check"), ptr(detail::mpi_data_.mpix_continuations_request)));

            using pika::threads::detail::polling_status;
            if (!detail::mpi_data_.mpix_ready_.load(std::memory_order_relaxed))
                return pika::threads::detail::polling_status::idle;
            //
            int flag = 0;
            if (!singlethreaded)
            {
                std::unique_lock lk(detail::mpi_data_.mpix_lock, std::try_to_lock);
                if (!lk.owns_lock()) { return polling_status::busy; }
                mpi_ext_continuation_result_check(MPI_Test(
                    &detail::mpi_data_.mpix_continuations_request, &flag, MPI_STATUS_IGNORE));
                if (flag != 0)
                {
                    restart_mpix();
                    return pika::threads::detail::polling_status::busy;
                }
            }
            else
            {
                // there should not be a lock here, but the mpix stuff fails without it
                mpi_ext_continuation_result_check(MPI_Test(
                    &detail::mpi_data_.mpix_continuations_request, &flag, MPI_STATUS_IGNORE));
                if (flag != 0)
                {
                    restart_mpix();
                    return pika::threads::detail::polling_status::busy;
                }
            }
            // for debugging, create a timer : debug info every N seconds
            static auto mpix_deb = mpi_debug<1>.make_timer(2, debug::detail::str<>("MPIX"), "poll");
            PIKA_DETAIL_DP(mpi_debug<1>,
                timed(mpix_deb, ptr(detail::mpi_data_.mpix_continuations_request), "Flag", flag));

            return polling_status::idle;
        }
#endif

        // -------------------------------------------------------------
        // Background progress function for MPI async operations
        // Checks for completed MPI_Requests and readies sender when complete
        pika::threads::detail::polling_status poll_multithreaded()
        {
            using pika::threads::detail::polling_status;

            // ---------------
            // If a thread is already polling and found a completion,
            // it first places it on the ready requests queue and any thread
            // can invoke the callback without being under lock
            ready_callback ready_callback_;
            while (mpi_data_.ready_requests_.try_dequeue(ready_callback_))
            {
#ifdef PIKA_HAVE_APEX
                apex::scoped_timer apex_invoke("pika::mpi::trigger");
#endif
                PIKA_DETAIL_DP(mpi_debug<4>,
                    debug(str<>("Ready CB invoke"), ptr(ready_callback_.request_),
                        ready_callback_.err_));

                // decrement before invoking callback : race if invoked code checks in_flight
                --mpi_data_.all_in_flight_;
                PIKA_INVOKE(PIKA_MOVE(ready_callback_.cb_), ready_callback_.err_);
                pika::threads::detail::decrement_global_activity_count();
            }

            // if we think there are no outstanding requests, then exit quickly
            if (mpi_data_.all_in_flight_.load(std::memory_order_relaxed) == 0)
                return polling_status::idle;

            // start a scoped block where the polling lock is held
            {
#ifdef PIKA_HAVE_APEX
                //apex::scoped_timer apex_poll("pika::mpi::poll");
#endif
                std::unique_lock<mutex_type> lk(mpi_data_.polling_vector_mtx_, std::try_to_lock);
                if (!lk.owns_lock())
                {
                    if constexpr (mpi_debug<5>.is_enabled())
                    {
                        // for debugging, create a timer : debug info every N seconds
                        static auto poll_deb =
                            mpi_debug<5>.make_timer(2, debug::detail::str<>("Poll - lock failed"));
                        PIKA_DETAIL_DP(mpi_debug<5>, timed(poll_deb, mpi_data_));
                    }
                    return polling_status::idle;
                }

                if constexpr (mpi_debug<5>.is_enabled())
                {
                    // for debugging, create a timer : debug info every N seconds
                    static auto poll_deb =
                        mpi_debug<5>.make_timer(2, debug::detail::str<>("Poll - lock success"));
                    PIKA_DETAIL_DP(mpi_debug<5>, timed(poll_deb, mpi_data_));
                }

                bool event_handled;
                do {
                    event_handled = false;

                    // Move requests in the queue (that have not yet been polled for)
                    // into the polling vector ...
                    // Number in_flight does not change during this section as one
                    // is moved off the queue and into the vector
                    request_callback req_callback;
                    while (mpi_data_.request_callback_queue_.try_dequeue(req_callback))
                    {
                        add_to_request_callback_vector(PIKA_MOVE(req_callback));
                    }

                    std::uint32_t vsize = mpi_data_.requests_.size();

                    int num_completed = 0;
                    // do we poll for N requests at a time, or just 1
                    if (mpi_data_.max_polling_requests.load(std::memory_order_relaxed) > 1)
                    {
                        // it seems some MPI implementations choke when the request list is
                        // large, so we will use a max of max_poll_requests per test.
                        std::array<MPI_Status, max_poll_requests> status_vector_;
                        std::array<int, max_poll_requests> indices_vector_;

                        int req_init = 0;
                        while (vsize > 0)
                        {
                            int req_size = (std::min)(vsize, max_poll_requests);
                            /* @TODO: if we use MPI_STATUSES_IGNORE - how do we report failures? */
                            int status = MPI_Testsome(req_size, &mpi_data_.requests_[req_init],
                                &num_completed, indices_vector_.data(),
                                /*MPI_STATUSES_IGNORE*/ status_vector_.data());

                            // status field holds a valid error
                            bool status_valid = (status == MPI_ERR_IN_STATUS);
                            if (num_completed != MPI_UNDEFINED && num_completed > 0)
                            {
                                event_handled = true;
                                PIKA_DETAIL_DP(mpi_debug<4>,
                                    debug(str<>("MPI_Testsome"), mpi_data_, "num_completed",
                                        dec<3>(num_completed)));

                                // for each completed request
                                for (int i = 0; i < num_completed; ++i)
                                {
                                    size_t index = indices_vector_[i];
                                    mpi_data_.ready_requests_.enqueue(
                                        {PIKA_MOVE(mpi_data_.callbacks_[req_init + index].cb_),
                                            mpi_data_.callbacks_[req_init + index].request_,
                                            status_valid ? status_vector_[i].MPI_ERROR :
                                                           MPI_SUCCESS});
                                    // Remove the request from our vector to prevent retesting
                                    mpi_data_.requests_[req_init + index] = MPI_REQUEST_NULL;
                                }
                            }
                            vsize -= req_size;
                            req_init += req_size;
                        }
                    }
                    else
                    {
                        int rindex, flag;
                        int status = MPI_Testany(mpi_data_.requests_.size(),
                            mpi_data_.requests_.data(), &rindex, &flag, MPI_STATUS_IGNORE);
                        if (rindex != MPI_UNDEFINED)
                        {
                            size_t index = static_cast<size_t>(rindex);
                            event_handled = true;
                            mpi_data_.ready_requests_.enqueue(
                                {PIKA_MOVE(mpi_data_.callbacks_[index].cb_),
                                    mpi_data_.callbacks_[index].request_, status});
                            // Remove the request from our vector to prevent retesting
                            mpi_data_.requests_[index] = MPI_REQUEST_NULL;
                        }
                    }
                } while (event_handled == true);

                // still under lock : remove wasted space caused by completed requests
                compact_vectors();
            }    // end lock scope block

            // output a debug heartbeat every N seconds
            if constexpr (mpi_debug<4>.is_enabled())
            {
                static auto poll_deb =
                    mpi_debug<4>.make_timer(1, debug::detail::str<>("Poll - success"));
                PIKA_DETAIL_DP(mpi_debug<4>, timed(poll_deb, mpi_data_));
            }

            // invoke (new) ready callbacks without being under lock
            while (mpi_data_.ready_requests_.try_dequeue(ready_callback_))
            {
#ifdef PIKA_HAVE_APEX
                apex::scoped_timer apex_invoke("pika::mpi::trigger");
#endif
                PIKA_DETAIL_DP(mpi_debug<5>,
                    debug(str<>("CB invoke"), ptr(ready_callback_.request_), ready_callback_.err_));

                // decrement before invoking callback : race if invoked code checks in_flight
                --mpi_data_.all_in_flight_;
                PIKA_INVOKE(PIKA_MOVE(ready_callback_.cb_), ready_callback_.err_);
                pika::threads::detail::decrement_global_activity_count();
            }

            return mpi_data_.all_in_flight_.load(std::memory_order_relaxed) == 0 ?
                polling_status::idle :
                polling_status::busy;
        }

        // -------------------------------------------------------------
        // Background progress function for MPI async operations
        // Checks for completed MPI_Requests and readies sender when complete
        pika::threads::detail::polling_status poll_singlethreaded()
        {
            using pika::threads::detail::polling_status;

            // if we think there are no outstanding requests, then exit quickly
            if (mpi_data_.all_in_flight_.load(std::memory_order_relaxed) == 0)
                return polling_status::idle;

#ifdef PIKA_HAVE_APEX
                //apex::scoped_timer apex_poll("pika::mpi::poll");
#endif
            if constexpr (mpi_debug<5>.is_enabled())
            {
                // for debugging, create a timer : debug info every N seconds
                static auto poll_deb =
                    mpi_debug<5>.make_timer(2, debug::detail::str<>("Poll - lock success"));
                PIKA_DETAIL_DP(mpi_debug<5>, timed(poll_deb, mpi_data_));
            }

            bool event_handled;
            do {
                event_handled = false;

                // Move unpolled requests in the queue into the polling vector ...
                request_callback req_callback;
                while (mpi_data_.request_callback_queue_.try_dequeue(req_callback))
                {
                    add_to_request_callback_vector(PIKA_MOVE(req_callback));
                }

                int rindex, flag;
                int status = MPI_Testany(mpi_data_.requests_.size(), mpi_data_.requests_.data(),
                    &rindex, &flag, MPI_STATUS_IGNORE);
                if (rindex != MPI_UNDEFINED)
                {
                    size_t index = static_cast<size_t>(rindex);
                    event_handled = true;

                    PIKA_DETAIL_DP(mpi_debug<5>,
                        debug(
                            str<>("CB invoke"), ptr(mpi_data_.callbacks_[index].request_), status));

                    // Remove the request from our vector to prevent retesting
                    mpi_data_.requests_[index] = MPI_REQUEST_NULL;

                    // decrement before invoking callback : race if invoked code checks in_flight
                    --mpi_data_.all_in_flight_;
                    PIKA_INVOKE(PIKA_MOVE(mpi_data_.callbacks_[index].cb_), status);
                    pika::threads::detail::decrement_global_activity_count();
                }
            } while (event_handled == true);

            compact_vectors();

            // output a debug heartbeat every N seconds
            if constexpr (mpi_debug<4>.is_enabled())
            {
                static auto poll_deb =
                    mpi_debug<4>.make_timer(1, debug::detail::str<>("Poll - success"));
                PIKA_DETAIL_DP(mpi_debug<4>, timed(poll_deb, mpi_data_));
            }

            return mpi_data_.all_in_flight_.load(std::memory_order_relaxed) == 0 ?
                polling_status::idle :
                polling_status::busy;
        }

        // -------------------------------------------------------------
        inline bool singlethreaded(int mode)
        {
            return (use_pool(mode) && !use_inline_request(mode));
        }

        // -------------------------------------------------------------
        pika::threads::detail::polling_status poll()
        {
            // get mpi completion mode settings
            auto mode = get_completion_mode();
            bool single_threaded = singlethreaded(mode);

#ifdef OMPI_HAVE_MPI_EXT_CONTINUE
            // if mpi continuations are available, poll here and bypass main routine
            if (get_handler_method(mode) == handler_method::mpix_continuation)
            {
                return try_mpix_polling(single_threaded);
            }
#endif

            if (single_threaded) { return poll_singlethreaded(); }
            return poll_multithreaded();
        }

        // -------------------------------------------------------------
        void register_polling(pika::threads::detail::thread_pool_base& pool)
        {
#if defined(PIKA_DEBUG)
            ++get_register_polling_count();
#endif
            if (detail::mpi_data_.rank_ == 0)
            {
                PIKA_DETAIL_DP(detail::mpi_debug<1>,
                    debug(str<>("polling_enabled"), "pool =", pool.get_pool_name(), ", mode",
                        mode_string(get_completion_mode()), get_completion_mode()));
            }
            auto* sched = pool.get_scheduler();
            sched->set_mpi_polling_functions(&detail::poll, &get_work_count);
        }

        // -------------------------------------------------------------
        void unregister_polling(pika::threads::detail::thread_pool_base& pool)
        {
#if defined(PIKA_DEBUG)
            {
                std::unique_lock<detail::mutex_type> lk(detail::mpi_data_.polling_vector_mtx_);
                bool request_queue_empty =
                    (detail::mpi_data_.request_callback_queue_.size_approx() == 0);
                bool requests_empty = (detail::mpi_data_.all_in_flight_ == 0);
                lk.unlock();
                PIKA_ASSERT_MSG(request_queue_empty,
                    "MPI request polling was disabled while there are unprocessed MPI requests. "
                    "Make sure MPI request polling is not disabled too early.");
                PIKA_ASSERT_MSG(requests_empty,
                    "MPI request polling was disabled while there are active MPI futures. Make "
                    "sure MPI request polling is not disabled too early.");
            }
#endif
            PIKA_DETAIL_DP(mpi_debug<1>, debug(str<>("disable polling")));
            auto* sched = pool.get_scheduler();
            sched->clear_mpi_polling_function();
        }

        int comm_world_size() { return detail::mpi_data_.size_; }

#ifdef OMPI_HAVE_MPI_EXT_CONTINUE
        void register_mpix_continuation(
            MPI_Request* request, MPIX_Continue_cb_function* cb_func, void* op_state)
        {
            PIKA_DETAIL_DP(
                mpi_debug<2>, debug(str<>("MPIX"), "register continuation", ptr(request)));
            mpi_ext_continuation_result_check(MPIX_Continue(request, cb_func, op_state,
                /*MPIX_CONT_DEFER_COMPLETE | */ MPIX_CONT_INVOKE_FAILED, MPI_STATUSES_IGNORE,
                detail::mpi_data_.mpix_continuations_request));
        }

        void restart_mpix()
        {
            {
                PIKA_DETAIL_DP(mpi_debug<2>,
                    debug(str<>("MPIX"), "MPI_Start",
                        ptr(detail::mpi_data_.mpix_continuations_request)));
                mpi_ext_continuation_result_check(
                    MPI_Start(&detail::mpi_data_.mpix_continuations_request));
            }
        }
#else
        void register_mpix_continuation(MPI_Request*, MPIX_Continue_cb_function*, void*) {}
        void restart_mpix() {}
#endif

    }    // namespace detail

    // -------------------------------------------------------------
    size_t get_work_count() { return detail::mpi_data_.all_in_flight_; }

    // -------------------------------------------------------------
    void set_max_polling_size(std::size_t p) { detail::mpi_data_.max_polling_requests = p; }

    // -------------------------------------------------------------
    std::size_t get_max_polling_size()
    {
        return detail::mpi_data_.max_polling_requests.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------
    std::size_t get_completion_mode() { return detail::completion_flags_; }
    void set_completion_mode(std::size_t mode) { detail::completion_flags_ = mode; }

    // -----------------------------------------------------------------
    bool create_pool(std::string const& pool_name, pool_create_mode mode)
    {
        int is_initialized;
        MPI_Initialized(&is_initialized);
        if (is_initialized)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_data_.rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_data_.size_);
        }
        else
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "mpi::create_pool",
                "MPI must be initialized prior to pool creation");
        }
        // use reference so that we change the actual value in place
        std::size_t& flags = detail::completion_flags_;
        if (mode == pool_create_mode::force_create)
            flags |= 1;
        else if (mode == pool_create_mode::force_no_create)
            flags &= ~1;
        else if ((mode == pool_create_mode::pika_decides) && (detail::mpi_data_.size_ == 1))
        {
            // if we have a single rank - disable pool
            PIKA_DETAIL_DP(detail::mpi_debug<1>, debug(str<>("single rank"), "Pool disabled"));
            flags &= ~1;
        }
        PIKA_DETAIL_DP(detail::mpi_debug<1>,
            debug(str<>("completion mode"), bin<8>(flags), detail::mode_string(flags)));
        // override the variable used to control completion mode and pool flags
        setenv("PIKA_MPI_COMPLETION_MODE", std::to_string(flags).c_str(), true);

        // if pool is now disabled, just exit
        if (!detail::use_pool(flags))
        {
            set_pool_name("default");
            return false;
        }
        // Disable idle backoff on the MPI pool
        using pika::threads::scheduler_mode;
        auto smode = scheduler_mode::default_mode & ~scheduler_mode::enable_idle_backoff;

        // Create a thread pool with a single core that we will use for all
        // communication related tasks
        std::string name = pool_name;
        if (name.empty()) { name = get_pool_name(); }
        else
        {
            // override mpi pool name with whatever we decided on
            set_pool_name(name);
        }
        detail::pool_exists_ = true;
        //
        auto& rp = resource::get_partitioner();
        rp.create_thread_pool(
            get_pool_name(), pika::resource::scheduling_policy::static_priority, smode);
        rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0], get_pool_name());
        PIKA_DETAIL_DP(detail::mpi_debug<1>,
            debug(str<>("pool created"), "name", get_pool_name(), "mode flags", bin<8>(flags)));
        return true;
    }

    // -----------------------------------------------------------------
    const std::string& get_pool_name() { return detail::polling_pool_name_; }

    // -----------------------------------------------------------------
    void set_pool_name(const std::string& name) { detail::polling_pool_name_ = name; }

    // -----------------------------------------------------------------
    bool pool_exists() { return detail::pool_exists_; }

    // -----------------------------------------------------------------
    void register_polling()
    {
        auto mode = get_completion_mode();
        // enable pika polling on the mpi pool if the handling mode needs it
        if (detail::get_handler_method(mode) != detail::handler_method::yield_while)
        {
            PIKA_DETAIL_DP(detail::mpi_debug<1>,
                debug(
                    str<>("enabling polling"), "pool", get_pool_name(), detail::mode_string(mode)));
            detail::register_polling(pika::resource::get_thread_pool(get_pool_name()));
        }
    }

    // -------------------------------------------------------------
    void enable_optimizations(bool enable) { detail::mpi_data_.optimizations_ = enable; }

    // -------------------------------------------------------------
    // Return the "best" mpi thread mode to use for initialization
    // if all requests are transferred to the mpi pool, then single threaded is ok
    int get_preferred_thread_mode()
    {
        int required = MPI_THREAD_MULTIPLE;
        if (detail::singlethreaded(get_completion_mode()) && detail::mpi_data_.optimizations_)
        {
            required = MPI_THREAD_SINGLE;
            PIKA_DETAIL_DP(detail::mpi_debug<0>, debug(str<>("MPI_THREAD_SINGLE"), "overridden"));
        }
        return required;
    }

    // -----------------------------------------------------------------
    // initialize the pika::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    void init(bool init_mpi, bool init_errorhandler)
    {
        // don't allow polling code to run until init has completed
        std::lock_guard<detail::mutex_type> lk(detail::mpi_data_.polling_vector_mtx_);

        // --------------------------------------
        // the user has asked us to call mpi_init
        if (init_mpi)
        {
            int provided, required = get_preferred_thread_mode();
            pika::util::mpi_environment::init(nullptr, nullptr, required, required, provided);
            MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_data_.rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_data_.size_);
        }
        if (init_mpi)
        {
            int provided, required = MPI_THREAD_MULTIPLE;
            pika::util::mpi_environment::init(nullptr, nullptr, required, required, provided);
            MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_data_.rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_data_.size_);
        }
        else
        {
            int is_initialized = 0;
            MPI_Initialized(&is_initialized);
            if (is_initialized)
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_data_.rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_data_.size_);
            }
            else { PIKA_DETAIL_DP(detail::mpi_debug<1>, error(str<>("mpi not initialized"))); }
        }

        PIKA_DETAIL_DP(detail::mpi_debug<1>, debug(str<>("init"), detail::mpi_data_));

        // --------------------------------------
        // install error handler (convert mpi errors into exceptoions
        if (init_errorhandler)
        {
            detail::set_error_handler();
            detail::mpi_data_.error_handler_initialized_ = true;
        }

        // --------------------------------------
        auto mode = get_completion_mode();
        if (mode >= pika::detail::to_underlying(detail::handler_method::unspecified))
        {
            PIKA_THROW_EXCEPTION(
                pika::error::invalid_status, "Bad completion flags", "invalid completion mode");
        }
#ifdef OMPI_HAVE_MPI_EXT_CONTINUE
        // if we are using experimental mpix_continuations, setup internals
        if (detail::get_handler_method(mode) == detail::handler_method::mpix_continuation)
        {
            // the lock prevents multithreaded polling from accessing the request before it is ready
            std::unique_lock lk(detail::mpi_data_.mpix_lock);
            // the atomic flag prevents lockless version accessing the request before it is ready
            detail::mpi_ext_continuation_result_check(MPIX_Continue_init(MPIX_CONT_POLL_ONLY,
                MPI_UNDEFINED, MPI_INFO_NULL, &detail::mpi_data_.mpix_continuations_request));

            PIKA_DETAIL_DP(detail::mpi_debug<1>,
                debug(str<>("MPIX"), "Enable", "Pool", get_pool_name(), "Mode",
                    detail::mode_string(mode), ptr(detail::mpi_data_.mpix_continuations_request)));
            // it is now safe to use the mpix request, {memory_order = not a critical code path}
            detail::mpi_data_.mpix_ready_.store(true, std::memory_order_seq_cst);
        }
#else
        // if selected, but unsupported, throw an error
        if (detail::get_handler_method(mode) == detail::handler_method::mpix_continuation)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "MPI_EXT_CONTINUE",
                "mpi_ext_continuation not supported, invalid handler method");
        }
#endif
    }

    // -----------------------------------------------------------------
    void finalize(std::string const& pool_name)
    {
        if (detail::mpi_data_.error_handler_initialized_)
        {
            PIKA_ASSERT(detail::pika_mpi_errhandler != 0);
            detail::mpi_data_.error_handler_initialized_ = false;
            MPI_Errhandler_free(&detail::pika_mpi_errhandler);
            detail::pika_mpi_errhandler = 0;
        }

        // clean up if we initialized mpi
        pika::util::mpi_environment::finalize();

        PIKA_DETAIL_DP(detail::mpi_debug<5>,
            debug(str<>("Clearing mode"), detail::mpi_data_, "disable_user_polling"));

        if (pool_name.empty()) { detail::unregister_polling(pika::resource::get_thread_pool(0)); }
        else { detail::unregister_polling(pika::resource::get_thread_pool(pool_name)); }
    }
}    // namespace pika::mpi::experimental
