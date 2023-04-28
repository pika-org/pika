//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_mpi/mpi_exception.hpp>
#include <pika/async_mpi/mpi_polling.hpp>
#include <pika/command_line_handling/get_env_var.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/datastructures/detail/small_vector.hpp>
#include <pika/debugging/print.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/mpi_base/mpi_environment.hpp>
#include <pika/synchronization/condition_variable.hpp>
#include <pika/synchronization/counting_semaphore.hpp>
#include <pika/synchronization/mutex.hpp>

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

    constexpr std::uint32_t max_mpi_streams = static_cast<std::uint32_t>(stream_type::max_stream);

    namespace detail {
        // -----------------------------------------------------------------
        // by convention the title is 7 chars (for alignment)
        // a debug level of N shows messages with level 1..N
        using namespace pika::debug::detail;
        template <int Level>
        static print_threshold<Level, 0> mpi_debug("MPIPOLL");

        // -----------------------------------------------------------------
        /// Queries an environment variable to get/override a default value for
        /// the number of messages allowed 'in flight' before we throttle a
        /// thread trying to send more data
        std::uint32_t get_throttling_default();
        std::size_t get_polling_default();
        std::size_t get_completion_mode_default();

        // -----------------------------------------------------------------
        /// Holds an MPI_Request and a callback. The callback is intended to be
        /// called when the operation tied to the request handle completes.
        struct request_callback
        {
            MPI_Request request_;
            request_callback_function_type callback_function_;
            stream_type index_{stream_type::automatic};
        };

        // -----------------------------------------------------------------
        /// To enable independent throttling of sends/receives/other
        /// we maintain several "queues" which have their own condition
        /// variables for suspension
        struct mpi_stream
        {
            pika::counting_semaphore<> semaphore_{get_throttling_default()};
            std::atomic<std::uint32_t> active_requests_{0};
            std::uint32_t limit_{get_throttling_default()};
#ifdef PIKA_DEBUG
            std::uint32_t index_;
#endif
        };

        struct mpi_callback_info
        {
            request_callback_function_type cb_;
            std::int32_t err_;
            mpi_stream* stream_;
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
        using mutex_type = pika::spinlock;

        PIKA_EXPORT const char* stream_name(stream_type s)
        {
            using namespace pika::mpi::experimental;
            switch (s)
            {
            case stream_type::automatic:
                return "auto";
                break;
            case stream_type::send_1:
                return "send_1";
                break;
            case stream_type::send_2:
                return "send_2";
                break;
            case stream_type::receive_1:
                return "recv_1";
                break;
            case stream_type::receive_2:
                return "recv_2";
                break;
            case stream_type::collective_1:
                return "coll_1";
                break;
            case stream_type::collective_2:
                return "coll_2";
                break;
            case stream_type::user_1:
                return "user_1";
                break;
            case stream_type::user_2:
                return "user_2";
                break;
            case stream_type::max_stream:
                return "smax";
                break;
            default:
                return "error";
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
            std::atomic<std::uint32_t> active_request_vector_size_{0};
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
            std::vector<MPI_Request> request_vector_;
            std::vector<mpi_callback_info> callback_vector_;
            //
            std::vector<MPI_Status> status_vector_;
            std::vector<int> indices_vector_;

            // mutex needed to protect mpi request vector, note that the
            // mpi poll function usually takes place inside the main scheduling loop
            // though poll may also be called directly by a user task.
            // we use a spinlock for both cases
            mutex_type polling_vector_mtx_;

            // streams used when throttling mpi traffic,
            std::array<mpi_stream, max_mpi_streams> default_queues_;
        };

        /// a single instance of all the mpi variables initialized once at startup
        static mpi_data mpi_data_;

        // default transfer mode for mpi continuations
        static std::size_t task_completion_mode_ = get_completion_mode_default();

        static std::string polling_pool_name_ = "pika:polling";
        static bool pool_exists_ = false;

        inline mpi_stream& get_stream_ref(stream_type stream)
        {
            return mpi_data_.default_queues_[static_cast<uint32_t>(stream)];
        }

        // stream operator to display debug mpi_data
        PIKA_EXPORT std::ostream& operator<<(std::ostream& os, mpi_data const& info)
        {
            // clang-format off
            os << "R "
               << dec<3>(info.rank_) << "/"
               << dec<3>(info.size_)
               << " vector "        << dec<4>(info.active_request_vector_size_)
               << " queued "        << dec<4>(info.request_queue_size_)
               << " all_in_flight " << dec<4>(info.all_in_flight_)
               << " vec_cb "        << dec<4>(info.callback_vector_.size())
               << " vec_rq "        << dec<4>(info.request_vector_.size());
            // clang-format on
            return os;
        }

        // stream operator to display debug mpi_stream
        PIKA_EXPORT std::ostream& operator<<(std::ostream& os, mpi_stream const& stream)
        {
            // clang-format off
            os
#ifdef PIKA_DEBUG
//               <<  "stream "    << stream_name(stream_type(stream.index_))
#endif
               << " in_stream " << dec<4>(stream.active_requests_)
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
            std::vector<MPI_Request>& vec = mpi_data_.request_vector_;
            return std::count_if(
                vec.begin(), vec.end(), [](MPI_Request r) { return r == MPI_REQUEST_NULL; });
        }

        // -----------------------------------------------------------------
        void wait_for_throttling_impl(mpi_stream& stream)
        {
            [[maybe_unused]] auto scp = mpi_debug<5>.scope("throttling", "wait", stream);

            stream.semaphore_.acquire();
            //            if (stream.active_requests_ < stream.limit_)
            //            {
            //                return;
            //            }
            // we don't bother with a condition/predicate, because it would be racy,
            // (any thread can post more messages between when we are woken and
            // when we test the "in_flight" condition)
            // and if we have any spurious wakeup, we don't really care as it just
            // means an extra message would be posted.
            // note that since we don't use a predicate, we use notify_one
            // and not notify_all to wake threads - if we used notify_all, then all
            // threads would always be woken and throttling would be compromised
            //
            // we do not throttle if we are not on a pika thread
            //            using namespace pika::threads::detail;
            //            if (get_self_id() != invalid_thread_id)
            //            {

            //                std::unique_lock lk(stream.throttling_mtx_);
            //                [[maybe_unused]] auto scp = mpi_debug<1>.scope("throttling", "wait", "locked", stream);
            //                stream.throttling_cond_.wait(lk);
            //            }
        }

        //        // -----------------------------------------------------------------
        //        void wait_for_throttling(stream_type stream)
        //        {
        //            // if throttling is disabled, then do nothing
        //            if constexpr (detail::throttling_enabled)
        //            {
        //                wait_for_throttling_impl(get_stream_ref(stream));
        //            }
        //        }

        // -----------------------------------------------------------------
        std::uint32_t get_throttling_default()
        {
            // if the global throttling var is set, set all streams
            std::uint32_t def = std::uint32_t(-1);    // unlimited
            std::uint32_t val = pika::get_env_value("PIKA_MPI_MSG_THROTTLE", def);
            for (size_t i = 0; i < mpi_data_.default_queues_.size(); ++i)
            {
                mpi_data_.default_queues_[i].limit_ = val;
                // mpi_data_.default_queues_[i].semaphore_ = std::move(pika::counting_semaphore<>{val});
#ifdef PIKA_DEBUG
                mpi_data_.default_queues_[i].index_ = i;
#endif
            }
            // check env settings for individual streams
            for (size_t i = 0; i < mpi_data_.default_queues_.size(); ++i)
            {
                std::string str =
                    "PIKA_MPI_MSG_THROTTLE_" + std::string(stream_name(stream_type(i)));
                std::transform(str.begin(), str.end(), str.begin(),
                    [](unsigned char c) { return std::toupper(c); });
                val = pika::get_env_value(str.c_str(), def);
                if (val != def)
                {
                    mpi_data_.default_queues_[i].limit_ = val;
                    // mpi_data_.default_queues_[i].semaphore_ = pika::counting_semaphore<>{val};
                }
            }
            // return default val for automatic (unspecified) stream
            return get_stream_ref(stream_type::automatic).limit_;
        }

        // -----------------------------------------------------------------
        std::size_t get_polling_default()
        {
            std::uint32_t val = pika::get_env_value("PIKA_MPI_POLLING_SIZE", 8);
            mpi_data_.max_polling_requests = val;
            return val;
        }

        // -----------------------------------------------------------------
        std::size_t get_completion_mode_default()
        {
            // inline continuations are default
            return pika::get_env_value("PIKA_MPI_COMPLETION_MODE", 1);
        }

        // -----------------------------------------------------------------
        /// used internally to add an MPI_Request to the lockfree queue
        /// that will be used by the polling routines to check when requests
        /// have completed
        void add_to_request_callback_queue(request_callback&& req_callback)
        {
            // access data before moving it
            auto stream = req_callback.index_;
            mpi_data_.request_callback_queue_.enqueue(PIKA_MOVE(req_callback));
            ++mpi_data_.request_queue_size_;
            ++get_stream_ref(stream).active_requests_;
            ++mpi_data_.all_in_flight_;
            PIKA_DETAIL_DP(mpi_debug<5>,
                debug(
                    str<>("CB queued"), req_callback.request_, get_stream_ref(stream), mpi_data_));
        }

        // -----------------------------------------------------------------
        /// used internally to add a request to the main polling vector
        /// that is passed to MPI_Testany. This is only called inside the
        /// polling function when a lock is held, so only one thread
        /// at a time ever enters here
        inline void add_to_request_callback_vector(request_callback&& req_callback)
        {
            mpi_data_.request_vector_.push_back(req_callback.request_);
            mpi_data_.callback_vector_.push_back({PIKA_MOVE(req_callback.callback_function_),
                MPI_SUCCESS, &get_stream_ref(req_callback.index_), req_callback.request_});
            ++(mpi_data_.active_request_vector_size_);

            // clang-format off
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("CB queue => vector"),
                mpi_data_, req_callback.request_, get_stream_ref(req_callback.index_),
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
        bool add_request_callback(request_callback_function_type&& callback, MPI_Request request,
            check_request_eager eager, stream_type s)
        {
            PIKA_ASSERT_MSG(get_register_polling_count() != 0,
                "MPI event polling has not been enabled on any pool. Make sure that MPI event "
                "polling is enabled on at least one thread pool.");

            // if already complete, skip callback
#ifndef DISALLOW_EAGER_POLLING_CHECK
            if (eager == check_request_eager::yes && detail::poll_request(request))
            {
                PIKA_DETAIL_DP(
                    mpi_debug<5>, debug(str<>("eager poll"), request, get_stream_ref(s)));
                // invoke the callback now since request has completed eagerly
                PIKA_INVOKE(PIKA_MOVE(callback), MPI_SUCCESS);
                // didn't increment 'in flight' counter, don't notify condition
                return false;
            }
#endif
            add_to_request_callback_queue(request_callback{request, PIKA_MOVE(callback), s});
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

        bool poll_request(MPI_Request& req)
        {
            int flag;
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            if (flag)
            {
                PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("eager poll ok"), req));
            }
            return flag;
        };

        // -------------------------------------------------------------
        /// Remove all entries in request and callback vectors that are invalid
        /// Ideally, would use a zip iterator to do both using remove_if
        void compact_vectors()
        {
            size_t const size = detail::mpi_data_.request_vector_.size();
            size_t pos = size;
            // find index of first NULL request
            for (size_t i = 0; i < size; ++i)
            {
                if (detail::mpi_data_.request_vector_[i] == MPI_REQUEST_NULL)
                {
                    pos = i;
                    break;
                }
            }
            // move all non NULL requests/callbacks towards beginning of vector.
            for (size_t i = pos + 1; i < size; ++i)
            {
                if (detail::mpi_data_.request_vector_[i] != MPI_REQUEST_NULL)
                {
                    detail::mpi_data_.request_vector_[pos] = detail::mpi_data_.request_vector_[i];
                    detail::mpi_data_.callback_vector_[pos] =
                        PIKA_MOVE(detail::mpi_data_.callback_vector_[i]);
                    pos++;
                }
            }
            // and trim off the space we didn't need
            detail::mpi_data_.request_vector_.resize(pos);
            detail::mpi_data_.callback_vector_.resize(pos);
        }

        // -------------------------------------------------------------
        struct mpi_test_helper
        {
            mpi_test_helper() {}
        };

        // -------------------------------------------------------------
        // Background progress function for MPI async operations
        // Checks for completed MPI_Requests and sets mpi::experimental::future
        // ready when found
        pika::threads::detail::polling_status poll()
        {
            using pika::threads::detail::polling_status;

            // invoke ready callbacks without being under lock
            detail::ready_callback ready_callback_;
            while (detail::mpi_data_.ready_requests_.try_dequeue(ready_callback_))
            {
                PIKA_DETAIL_DP(mpi_debug<5>,
                    debug(str<>("CB invoke"), ready_callback_.request_, ready_callback_.err_));

                // Invoke callback (PIKA_MOVE doesn't compile here)
                PIKA_INVOKE(std::move(ready_callback_.cb_), ready_callback_.err_);
            }

            if (detail::mpi_data_.all_in_flight_.load(std::memory_order_relaxed) == 0)
                return polling_status::idle;

            // start a scope block where the polling lock is held
            {
                std::unique_lock<detail::mutex_type> lk(
                    detail::mpi_data_.polling_vector_mtx_, std::try_to_lock);
                if (!lk.owns_lock())
                {
                    if constexpr (mpi_debug<5>.is_enabled())
                    {
                        // for debugging, create a timer : debug info every N seconds
                        static auto poll_deb =
                            mpi_debug<5>.make_timer(1, str<>("Poll - lock failed"));
                        PIKA_DETAIL_DP(mpi_debug<5>, timed(poll_deb, mpi_data_));
                    }
                    return polling_status::idle;
                }

                if constexpr (mpi_debug<5>.is_enabled())
                {
                    // for debugging, create a timer : debug info every N seconds
                    static auto poll_deb = mpi_debug<5>.make_timer(1, str<>("Poll - lock success"));
                    PIKA_DETAIL_DP(mpi_debug<5>, timed(poll_deb, mpi_data_));
                }

                bool event_handled;
                do
                {
                    event_handled = false;

                    // Move requests in the queue (that have not yet been polled for)
                    // into the polling vector ...
                    // Number in_flight does not change during this section as one
                    // is moved off the queue and into the vector
                    detail::request_callback req_callback;
                    while (detail::mpi_data_.request_callback_queue_.try_dequeue(req_callback))
                    {
                        --detail::mpi_data_.request_queue_size_;
                        add_to_request_callback_vector(PIKA_MOVE(req_callback));
                    }

                    const std::size_t max_test_vector_size = 512;
                    std::size_t vsize =
                        std::min(detail::mpi_data_.request_vector_.size(), max_test_vector_size);

                    int outcount = 0;
                    // do we poll for N requests at a time, or just 1
                    if (detail::mpi_data_.max_polling_requests > 1)
                    {
                        std::size_t req_size =
                            std::min(vsize, detail::mpi_data_.max_polling_requests);
                        detail::mpi_data_.indices_vector_.resize(req_size);
                        detail::mpi_data_.status_vector_.resize(req_size);

                        int result =
                            MPI_Testsome(req_size, detail::mpi_data_.request_vector_.data(),
                                &outcount, detail::mpi_data_.indices_vector_.data(),
                                detail::mpi_data_.status_vector_.data());
                        /*use MPI_STATUSES_IGNORE ?*/

                        if (result != MPI_SUCCESS)
                            throw mpi_exception(result, "MPI_Testsome error");
                        if (outcount != MPI_UNDEFINED && outcount != 0)
                        {
                            event_handled = true;
                            PIKA_DETAIL_DP(mpi_debug<5>,
                                debug(str<>("Polling loop"), detail::mpi_data_, "outcount",
                                    dec<3>(outcount)));

                            // for each completed request
                            for (int i = 0; i < outcount; ++i)
                            {
                                size_t index = detail::mpi_data_.indices_vector_[i];
                                detail::mpi_data_.ready_requests_.enqueue(
                                    {PIKA_MOVE(detail::mpi_data_.callback_vector_[index].cb_),
                                        detail::mpi_data_.callback_vector_[index].request_,
                                        detail::mpi_data_.status_vector_[i].MPI_ERROR});
                                // Remove the request from our vector to prevent retesting
                                detail::mpi_data_.request_vector_[index] = MPI_REQUEST_NULL;

                                // decrement before invoking callback to avoid race
                                // if invoked code checks in_flight value
                                --mpi_data_.all_in_flight_;
                                --mpi_data_.active_request_vector_size_;

                                //                                // wake any thread that is waiting for throttling
                                //                                if constexpr (detail::throttling_enabled)
                                //                                {
                                //                                    mpi_stream& stream =
                                //                                        *detail::mpi_data_.callback_vector_[index].stream_;
                                //                                    --stream.active_requests_;
                                //                                    stream.semaphore_.release();
                                //                                }
                            }
                        }
                    }
                    else
                    {
                        int rindex, flag;
                        int result = MPI_Testany(detail::mpi_data_.request_vector_.size(),
                            detail::mpi_data_.request_vector_.data(), &rindex, &flag,
                            MPI_STATUS_IGNORE);
                        if (result != MPI_SUCCESS)
                            throw mpi_exception(result, "MPI_Testany error");
                        if (rindex != MPI_UNDEFINED)
                        {
                            size_t index = static_cast<size_t>(rindex);
                            event_handled = true;
                            detail::mpi_data_.ready_requests_.enqueue(
                                {PIKA_MOVE(detail::mpi_data_.callback_vector_[index].cb_),
                                    detail::mpi_data_.callback_vector_[index].request_,
                                    MPI_SUCCESS});
                            // Remove the request from our vector to prevent retesting
                            detail::mpi_data_.request_vector_[index] = MPI_REQUEST_NULL;

                            // decrement before invoking callback to avoid race
                            // if invoked code checks in_flight value
                            --mpi_data_.all_in_flight_;
                            --mpi_data_.active_request_vector_size_;

                            // wake any thread that is waiting for throttling
                            if constexpr (detail::throttling_enabled)
                            {
                                mpi_stream& stream =
                                    *detail::mpi_data_.callback_vector_[index].stream_;
                                --stream.active_requests_;
                                stream.semaphore_.release();
                            }
                        }
                    }
                } while (event_handled == true);

                // still under lock : remove wasted space caused by completed requests
                compact_vectors();
            }    // end lock scope block

            // output a debug heartbeat every N seconds
            if constexpr (mpi_debug<5>.is_enabled())
            {
                static auto poll_deb = mpi_debug<5>.make_timer(1, str<>("Poll - success"));
                PIKA_DETAIL_DP(mpi_debug<5>, timed(poll_deb, mpi_data_));
            }

            // invoke (new) ready callbacks without being under lock
            while (detail::mpi_data_.ready_requests_.try_dequeue(ready_callback_))
            {
                PIKA_DETAIL_DP(mpi_debug<5>,
                    debug(str<>("CB invoke"), ready_callback_.request_, ready_callback_.err_));

                // Invoke callback (PIKA_MOVE doesn't compile here)
                PIKA_INVOKE(std::move(ready_callback_.cb_), ready_callback_.err_);
            }

            return detail::mpi_data_.all_in_flight_.load(std::memory_order_relaxed) == 0 ?
                polling_status::idle :
                polling_status::busy;
        }

        // -------------------------------------------------------------
        size_t get_work_count()
        {
            return mpi_data_.active_request_vector_size_ + mpi_data_.request_queue_size_;
        }

        // -------------------------------------------------------------
        void register_polling(pika::threads::detail::thread_pool_base& pool)
        {
#if defined(PIKA_DEBUG)
            ++get_register_polling_count();
#endif
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("enable polling")));
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
                bool request_vector_empty = detail::mpi_data_.all_in_flight_ == 0;
                lk.unlock();
                PIKA_ASSERT_MSG(request_queue_empty,
                    "MPI request polling was disabled while there are unprocessed MPI requests. "
                    "Make sure MPI request polling is not disabled too early.");
                PIKA_ASSERT_MSG(request_vector_empty,
                    "MPI request polling was disabled while there are active MPI futures. Make "
                    "sure MPI request polling is not disabled too early.");
            }
#endif
            PIKA_DETAIL_DP(mpi_debug<5>, debug(str<>("disable polling")));
            auto* sched = pool.get_scheduler();
            sched->clear_mpi_polling_function();
        }
    }    // namespace detail

    // -------------------------------------------------------------
    std::uint32_t set_max_requests_in_flight(std::uint32_t N, std::optional<stream_type> s)
    {
        throw std::runtime_error("Deprecated - @TODO semaphore disallows runtime changes");
        if (!s)
        {
            // start from 1
            for (size_t i = 1; i < detail::mpi_data_.default_queues_.size(); ++i)
            {
                detail::mpi_data_.default_queues_[i].limit_ = N;
                //detail::mpi_data_.default_queues_[i].semaphore_ = pika::counting_semaphore<>{N};
            }
            // set and return stream 0
            return std::exchange(detail::mpi_data_.default_queues_[0].limit_, N);
        }
        PIKA_ASSERT(
            static_cast<std::uint32_t>(s.value()) <= detail::mpi_data_.default_queues_.size());
        return std::exchange(detail::get_stream_ref(s.value()).limit_, N);
    }

    // -------------------------------------------------------------
    std::uint32_t get_max_requests_in_flight(std::optional<stream_type> s)
    {
        if (!s)
        {
            return detail::mpi_data_.default_queues_[0].limit_;
        }
        PIKA_ASSERT(
            static_cast<std::uint32_t>(s.value()) <= detail::mpi_data_.default_queues_.size());
        return detail::get_stream_ref(s.value()).limit_;
    }

    // -------------------------------------------------------------
    std::uint32_t get_num_requests_in_flight(stream_type s)
    {
        return detail::get_stream_ref(s).active_requests_;
    }

    // -------------------------------------------------------------
    void set_max_polling_size(std::size_t p)
    {
        detail::mpi_data_.max_polling_requests = p;
    }

    // -------------------------------------------------------------
    std::size_t get_max_polling_size()
    {
        return detail::mpi_data_.max_polling_requests;
    }

    // -----------------------------------------------------------------
    std::size_t get_completion_mode()
    {
        return detail::task_completion_mode_;
    }

    // -----------------------------------------------------------------
    const std::string& get_pool_name()
    {
        return detail::polling_pool_name_;
    }

    // -----------------------------------------------------------------
    void set_pool_name(const std::string& name)
    {
        detail::polling_pool_name_ = name;
    }

    // -----------------------------------------------------------------
    bool pool_exists()
    {
        return detail::pool_exists_;
    }

    // -------------------------------------------------------------
    // initialize the pika::mpi background request handler
    // All ranks should call this function,
    // but only one thread per rank needs to do so
    void init(bool init_mpi, std::string const& pool_name, bool init_errorhandler)
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

        PIKA_DETAIL_DP(detail::mpi_debug<1>,
            debug(str<>("pika::mpi::experimental::init"), "pool name", pool_name));

        // install polling loop on requested thread pool
        if (pool_name.empty() || !pika::resource::pool_exists(pool_name))
        {
            // does mpi pool exist with mpi pool name
            std::string name = mpi::experimental::get_pool_name();
            if (!pika::resource::pool_exists(name))
            {
                // drop back to default pika pool name
                name = resource::get_pool_name(0);
            }
            // override mpi pool name with whatever we decided on
            set_pool_name(name);
            auto mode = mpi::experimental::get_completion_mode();
            if (mode > 0 && mode < 100)
            {
                PIKA_DETAIL_DP(detail::mpi_debug<0>,
                    debug(str<>("register_polling"), name, "mode",
                        mpi::experimental::get_completion_mode()));
                detail::register_polling(pika::resource::get_thread_pool(name));
            }
            detail::pool_exists_ = (name != resource::get_pool_name(0));
        }
        else
        {
            PIKA_ASSERT_MSG(pool_name == get_pool_name(), "MPI pool name mismatch");
            // make sure the mpi pool name matches what the user passed in
            detail::pool_exists_ = true;
            set_pool_name(pool_name);
            auto mode = mpi::experimental::get_completion_mode();
            if (mode > 0 && mode < 100)
            {
                PIKA_DETAIL_DP(detail::mpi_debug<0>,
                    debug(str<>("register_polling"), pool_name, "mode",
                        mpi::experimental::get_completion_mode()));
                detail::register_polling(pika::resource::get_thread_pool(pool_name));
            }
            detail::pool_exists_ = (pool_name != resource::get_pool_name(0));
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

        if (pool_name.empty())
        {
            detail::unregister_polling(pika::resource::get_thread_pool(0));
        }
        else
        {
            detail::unregister_polling(pika::resource::get_thread_pool(pool_name));
        }
    }
}    // namespace pika::mpi::experimental
