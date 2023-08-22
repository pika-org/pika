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
#include <pika/string_util/case_conv.hpp>
#include <pika/synchronization/condition_variable.hpp>
#include <pika/synchronization/mutex.hpp>
#include <pika/threading_base/detail/global_activity_count.hpp>
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

namespace pika::mpi::experimental {

    namespace detail {
        // -----------------------------------------------------------------
        // by convention the title is 7 chars (for alignment)
        // a debug level of N shows messages with level 1..N
        using namespace pika::debug::detail;
        template <int Level>
        static print_threshold<Level, 0> mpi_debug("MPIPOLL");

        constexpr std::uint32_t max_mpi_streams = detail::to_underlying(stream_type::max_stream);
        constexpr std::uint32_t max_poll_requests = 32;

        // -----------------------------------------------------------------
        /// Queries an environment variable to get/override a default value for
        /// the number of messages allowed 'in flight' before we throttle a
        /// thread trying to send more data
        void init_throttling_default();
        std::size_t get_polling_default();
        //std::size_t get_completion_mode_default();

        // -----------------------------------------------------------------
        /// Holds an MPI_Request and a callback. The callback is intended to be
        /// called when the operation tied to the request handle completes.
        struct request_callback
        {
            MPI_Request request_;
            request_callback_function_type callback_function_;
        };

        // -----------------------------------------------------------------
        /// To enable independent throttling of sends/receives/other
        /// we maintain several "queues" which have their own condition
        /// variables for suspension
        struct mpi_stream
        {
            std::int32_t limit_;
            const char* name_;
            std::shared_ptr<semaphore_type> semaphore_;
        };

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

        PIKA_EXPORT const char* stream_name(stream_type s)
        {
            using namespace pika::mpi::experimental;
            switch (s)
            {
            case stream_type::automatic: return "auto"; break;
            case stream_type::send_1: return "send_1"; break;
            case stream_type::send_2: return "send_2"; break;
            case stream_type::receive_1: return "recv_1"; break;
            case stream_type::receive_2: return "recv_2"; break;
            case stream_type::collective_1: return "coll_1"; break;
            case stream_type::collective_2: return "coll_2"; break;
            case stream_type::user_1: return "user_1"; break;
            case stream_type::user_2: return "user_2"; break;
            case stream_type::max_stream: return "smax"; break;
            default: return "error";
            }
        }

        // -----------------------------------------------------------------
        /// a convenience structure to hold state vars in one place
        struct mpi_data
        {
            bool error_handler_initialized_ = false;
            int rank_ = -1;
            int size_ = -1;
            std::size_t max_polling_requests = get_polling_default();

            // requests vector holds the requests that are checked; this
            // represents the number of active requests in the vector, not the
            // size of the vector
            std::atomic<std::uint32_t> active_requests_size_{0};
            // requests queue holds the requests recently added
            std::atomic<std::uint32_t> request_queue_size_{0};
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

            // streams used when throttling mpi traffic,
            std::array<mpi_stream, max_mpi_streams> default_queues_;
        };

        struct initializer
        {
            initializer() { init_throttling_default(); }
        };

        /// a single instance of all the mpi variables initialized once at startup
        static mpi_data mpi_data_;

        void init_stream(stream_type s, std::int32_t limit)
        {
            mpi_stream& stream = mpi_data_.default_queues_[detail::to_underlying(s)];
            stream.semaphore_ = std::make_shared<semaphore_type>(limit);
            stream.name_ = stream_name(s);
            stream.limit_ = limit;
            PIKA_DETAIL_DP(mpi_debug<3>, debug(str<>("init stream"), stream.name_, dec<5>(limit)));
        }

        std::shared_ptr<semaphore_type> get_semaphore(stream_type s)
        {
            mpi_stream& stream = mpi_data_.default_queues_[detail::to_underlying(s)];
            PIKA_DETAIL_DP(mpi_debug<3>,
                debug(str<>("get stream"), stream.name_, dec<5>(stream.semaphore_.use_count())));
            return stream.semaphore_;
        }

        // default transfer mode for mpi continuations
        static std::size_t task_completion_flags_ = get_completion_mode_default();

        static std::string polling_pool_name_ = "polling";
        static bool pool_exists_ = false;

        inline mpi_stream& get_stream_ref(stream_type stream)
        {
            return mpi_data_.default_queues_[detail::to_underlying(stream)];
        }

        // stream operator to display debug mpi_data
        PIKA_EXPORT std::ostream& operator<<(std::ostream& os, mpi_data const& info)
        {
            // clang-format off
            os << "R "
               << dec<3>(info.rank_) << "/"
               << dec<3>(info.size_)
               << " vector "    << dec<4>(info.active_requests_size_)
               << " queued "    << dec<4>(info.request_queue_size_)
               << " in_flight " << dec<4>(info.all_in_flight_)
               << " vec_cb "    << dec<4>(info.callbacks_.size())
               << " vec_rq "    << dec<4>(info.requests_.size());
            // clang-format on
            return os;
        }

        // stream operator to display debug mpi_stream
        PIKA_EXPORT std::ostream& operator<<(std::ostream& os, mpi_stream const& stream)
        {
            // clang-format off
            os
               <<  "stream "    << stream.name_
               << " limit "     << dec<4>(stream.limit_);
            // clang-format on
            return os;
        }

        // stream operator to display debug mpi_request
        PIKA_EXPORT std::ostream& operator<<(std::ostream& os, MPI_Request const& req)
        {
            // clang-format off
            os <<  "req " << hex<8>(req);
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
        void init_throttling_default()
        {
            // if the global throttling var is set, set all streams
            std::uint32_t def = std::uint32_t(-1);    // unlimited
            std::uint32_t val =
                pika::detail::get_env_var_as<std::uint32_t>("PIKA_MPI_MSG_THROTTLE", def);
            for (size_t i = 0; i < mpi_data_.default_queues_.size(); ++i)
            {
                detail::init_stream(stream_type(i), val);
            }
            // check env settings for individual streams
            for (size_t i = 0; i < mpi_data_.default_queues_.size(); ++i)
            {
                std::string str =
                    "PIKA_MPI_MSG_THROTTLE_" + std::string(stream_name(stream_type(i)));
                pika::detail::to_upper(str);
                val = pika::detail::get_env_var_as<std::uint32_t>(str.c_str(), def);
                if (val != def) { detail::init_stream(stream_type(i), val); }
            }
        }

        // -----------------------------------------------------------------
        std::size_t get_polling_default()
        {
            std::uint32_t val =
                pika::detail::get_env_var_as<std::uint32_t>("PIKA_MPI_POLLING_SIZE", 8);
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("Poll size"), dec<3>(val)));
            mpi_data_.max_polling_requests = val;
            return val;
        }

        // -----------------------------------------------------------------
        std::size_t get_completion_mode_default()
        {
            // inline continuations are default
            task_completion_flags_ =
                pika::detail::get_env_var_as<std::size_t>("PIKA_MPI_COMPLETION_MODE", 1);
            return task_completion_flags_;
        }

        // -----------------------------------------------------------------
        /// used internally to add an MPI_Request to the lockfree queue
        /// that will be used by the polling routines to check when requests
        /// have completed
        void add_to_request_callback_queue(request_callback&& req_callback)
        {
            pika::threads::detail::increment_global_activity_count();

            // access data before moving it
            mpi_data_.request_callback_queue_.enqueue(PIKA_MOVE(req_callback));
            ++mpi_data_.request_queue_size_;
            ++mpi_data_.all_in_flight_;
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
            ++(mpi_data_.active_requests_size_);

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
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("pika_MPI_Handler")));
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
            if (flag) { PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("poll MPI_Test ok"), ptr(req))); }
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

        // -------------------------------------------------------------
        // Background progress function for MPI async operations
        // Checks for completed MPI_Requests and sets mpi::experimental::future
        // ready when found
        pika::threads::detail::polling_status poll()
        {
            using pika::threads::detail::polling_status;
            using namespace pika::mpi::experimental::detail;

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

                // Invoke callback (PIKA_MOVE doesn't compile here)
                PIKA_INVOKE(std::move(ready_callback_.cb_), ready_callback_.err_);
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
                            mpi_debug<5>.make_timer(2, str<>("Poll - lock failed"));
                        PIKA_DETAIL_DP(mpi_debug<5>, timed(poll_deb, mpi_data_));
                    }
                    return polling_status::idle;
                }

                if constexpr (mpi_debug<5>.is_enabled())
                {
                    // for debugging, create a timer : debug info every N seconds
                    static auto poll_deb = mpi_debug<5>.make_timer(2, str<>("Poll - lock success"));
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
                        --mpi_data_.request_queue_size_;
                    }

                    std::uint32_t vsize = mpi_data_.requests_.size();

                    int num_completed = 0;
                    // do we poll for N requests at a time, or just 1
                    if (mpi_data_.max_polling_requests > 1)
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

                                    // decrement before invoking callback to avoid race
                                    // if invoked code checks in_flight value
                                    --mpi_data_.all_in_flight_;
                                    --mpi_data_.active_requests_size_;
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

                            // decrement before invoking callback to avoid race
                            // if invoked code checks in_flight value
                            --mpi_data_.all_in_flight_;
                            --mpi_data_.active_requests_size_;
                        }
                    }
                } while (event_handled == true);

                // still under lock : remove wasted space caused by completed requests
                compact_vectors();
            }    // end lock scope block

            // output a debug heartbeat every N seconds
            if constexpr (mpi_debug<4>.is_enabled())
            {
                static auto poll_deb = mpi_debug<4>.make_timer(1, str<>("Poll - success"));
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

                // Invoke callback (PIKA_MOVE doesn't compile here)
                PIKA_INVOKE(std::move(ready_callback_.cb_), ready_callback_.err_);

                pika::threads::detail::decrement_global_activity_count();
            }

            return mpi_data_.all_in_flight_.load(std::memory_order_relaxed) == 0 ?
                polling_status::idle :
                polling_status::busy;
        }

        // -------------------------------------------------------------
        void register_polling(pika::threads::detail::thread_pool_base& pool)
        {
#if defined(PIKA_DEBUG)
            ++get_register_polling_count();
#endif
            if (detail::mpi_data_.rank_ == 0)
            {
                PIKA_DETAIL_DP(detail::mpi_debug<2>,
                    debug(str<>("register_polling"), pool.get_pool_name(), "mode",
                        mpi::experimental::get_completion_mode()));
            }
            auto* sched = pool.get_scheduler();
            sched->set_mpi_polling_functions(
                &pika::mpi::experimental::detail::poll, &get_work_count);
        }

        // -------------------------------------------------------------
        void unregister_polling(pika::threads::detail::thread_pool_base& pool)
        {
#if defined(PIKA_DEBUG)
            {
                std::unique_lock<pika::mpi::experimental::detail::mutex_type> lk(
                    detail::mpi_data_.polling_vector_mtx_);
                bool request_queue_empty =
                    detail::mpi_data_.request_callback_queue_.size_approx() == 0;
                bool requests_empty = detail::mpi_data_.all_in_flight_ == 0;
                lk.unlock();
                PIKA_ASSERT_MSG(request_queue_empty,
                    "MPI request polling was disabled while there are unprocessed MPI requests. "
                    "Make sure MPI request polling is not disabled too early.");
                PIKA_ASSERT_MSG(requests_empty,
                    "MPI request polling was disabled while there are active MPI futures. Make "
                    "sure MPI request polling is not disabled too early.");
            }
#endif
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("disable polling")));
            auto* sched = pool.get_scheduler();
            sched->clear_mpi_polling_function();
        }

        int comm_world_size() { return detail::mpi_data_.size_; }

    }    // namespace detail

    // -------------------------------------------------------------
    size_t get_work_count()
    {
        return detail::mpi_data_.active_requests_size_ + detail::mpi_data_.request_queue_size_;
    }

    // -------------------------------------------------------------
    void set_max_requests_in_flight(std::uint32_t N, std::optional<stream_type> s)
    {
        if (!s)
        {
            // start from 1
            for (size_t i = 1; i < detail::mpi_data_.default_queues_.size(); ++i)
            {
                std::cout << "Here 1" << std::endl;
                detail::init_stream(stream_type(i), N);
            }
        }
        else
        {
            PIKA_ASSERT(
                detail::to_underlying(s.value()) <= detail::mpi_data_.default_queues_.size());
            detail::init_stream(s.value(), N);
        }
    }

    // -------------------------------------------------------------
    std::uint32_t get_max_requests_in_flight(std::optional<stream_type> s)
    {
        if (!s) { return detail::mpi_data_.default_queues_[0].limit_; }
        PIKA_ASSERT(detail::to_underlying(s.value()) <= detail::mpi_data_.default_queues_.size());
        return detail::get_stream_ref(s.value()).limit_;
    }

    // -------------------------------------------------------------
    void set_max_polling_size(std::size_t p) { detail::mpi_data_.max_polling_requests = p; }

    // -------------------------------------------------------------
    std::size_t get_max_polling_size() { return detail::mpi_data_.max_polling_requests; }

    // -----------------------------------------------------------------
    std::size_t get_completion_mode() { return detail::task_completion_flags_; }

    // -----------------------------------------------------------------
    bool create_pool(
        pika::resource::partitioner& rp, std::string const& pool_name, pool_create_mode mode)
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
        //
        int flags = detail::get_completion_mode_default();
        if (mode == pool_create_mode::force_create)
            flags |= 1;
        else if (mode == pool_create_mode::force_no_create)
            flags &= ~1;
        else if (mode == pool_create_mode::pika_decides)
        {
            // if we have a single rank - disable pool
            if (detail::mpi_data_.size_ == 1) { flags &= ~1; }
        }
        using namespace pika::debug::detail;
        PIKA_DETAIL_DP(detail::mpi_debug<6>,
            debug(str<>("completion mode"), bin<8>(flags), detail::mode_string(flags)));
        // override the variable used to control completion mode and pool flags
        setenv("PIKA_MPI_COMPLETION_MODE", std::to_string(flags).c_str(), true);
        // and override the main flag
        detail::task_completion_flags_ = flags;

        // if pool is now disabled, just exit
        if ((flags & 1) == 0) return false;

        // Disable idle backoff on the MPI pool
        using pika::threads::scheduler_mode;
        auto smode = scheduler_mode::default_mode & ~scheduler_mode::enable_idle_backoff;

        // Create a thread pool with a single core that we will use for all
        // communication related tasks
        std::string name = pool_name;
        if (name.empty()) { name = mpi::experimental::get_pool_name(); }
        else
        {
            // override mpi pool name with whatever we decided on
            set_pool_name(name);
        }
        detail::pool_exists_ = true;
        //
        rp.create_thread_pool(
            get_pool_name(), pika::resource::scheduling_policy::static_priority, smode);
        rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0], get_pool_name());
        PIKA_DETAIL_DP(detail::mpi_debug<1>,
            debug(str<>("pool created"), "name", name, "mode flags", bin<8>(flags)));
        return true;
    }

    // -----------------------------------------------------------------
    const std::string& get_pool_name() { return detail::polling_pool_name_; }

    // -----------------------------------------------------------------
    void set_pool_name(const std::string& name) { detail::polling_pool_name_ = name; }

    // -----------------------------------------------------------------
    bool pool_exists() { return detail::pool_exists_; }

    // -------------------------------------------------------------
    // initialize the pika::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    void init(bool init_mpi, bool init_errorhandler)
    {
        using namespace pika::debug::detail;
        if (init_mpi)
        {
            int required = MPI_THREAD_MULTIPLE;
            int minimal = MPI_THREAD_FUNNELED;
            int provided;
            pika::util::mpi_environment::init(nullptr, nullptr, required, minimal, provided);
            if (provided < MPI_THREAD_FUNNELED)
            {
                PIKA_DETAIL_DP(detail::mpi_debug<5>,
                    error(str<>("pika::mpi::experimental::init"), "init failed"));
                PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::mpi::experimental::init",
                    "the MPI installation doesn't allow multiple threads");
            }
            MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_data_.rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_data_.size_);
        }
        else
        {
            // Check if MPI_Init has been called previously
            if (detail::mpi_data_.size_ == -1)
            {
                int is_initialized = 0;
                MPI_Initialized(&is_initialized);
                if (is_initialized)
                {
                    MPI_Comm_rank(MPI_COMM_WORLD, &detail::mpi_data_.rank_);
                    MPI_Comm_size(MPI_COMM_WORLD, &detail::mpi_data_.size_);
                }
            }
        }

        PIKA_DETAIL_DP(
            detail::mpi_debug<5>, debug(str<>("pika::mpi::experimental::init"), detail::mpi_data_));

        if (init_errorhandler)
        {
            detail::set_error_handler();
            detail::mpi_data_.error_handler_initialized_ = true;
        }

        std::string name = get_pool_name();
        // install polling loop on mpi thread pool
        if (!detail::pool_exists_)
        {
            // drop back to default pika pool name
            name = resource::get_pool_name(0);
        }
        // override mpi pool name with whatever we decided on
        set_pool_name(name);
        //
        auto mode = mpi::experimental::get_completion_mode();
        if (pika::mpi::experimental::detail::get_handler_mode(mode) !=
            detail::handler_mode::yield_while)
        {
            PIKA_DETAIL_DP(detail::mpi_debug<5>,
                debug(
                    str<>("enabling polling"), name, mpi::experimental::detail::mode_string(mode)));
            detail::register_polling(pika::resource::get_thread_pool(name));
        }

        static bool defaults_set = false;
        if (!defaults_set)
        {
            detail::init_throttling_default();
            defaults_set = true;
        }
    }

    // -----------------------------------------------------------------
    void finalize(std::string const& pool_name)
    {
        using namespace pika::debug::detail;
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
