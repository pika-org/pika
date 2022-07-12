//  Copyright (c) 2016 Zahra Khatami
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/persistent_auto_chunk_size.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/execution_base/traits/is_executor_parameters.hpp>
#include <pika/modules/timing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace pika { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces and then assigned to threads.
    /// The number of loop iterations combined is determined based on
    /// measurements of how long the execution of 1% of the overall number of
    /// iterations takes.
    /// This executor parameters type makes sure that as many loop iterations
    /// are combined as necessary to run for the amount of time specified.
    ///
    struct persistent_auto_chunk_size
    {
    public:
        /// Construct an \a persistent_auto_chunk_size executor parameters object
        ///
        /// \note Default constructed \a persistent_auto_chunk_size executor parameter
        ///       types will use 0 microseconds as the execution time for each chunk
        ///       and 80 microseconds as the minimal time for which
        ///       any of the scheduled chunks should run.
        ///
        constexpr persistent_auto_chunk_size(
            std::uint64_t num_iters_for_timing = 0)
          : chunk_size_time_(0)
          , min_time_(200000)
          , num_iters_for_timing_(num_iters_for_timing)
        {
        }

        /// Construct an \a persistent_auto_chunk_size executor parameters object
        ///
        /// \param time_cs      The execution time for each chunk.
        ///
        explicit persistent_auto_chunk_size(
            pika::chrono::steady_duration const& time_cs,
            std::uint64_t num_iters_for_timing = 0)
          : chunk_size_time_(time_cs.value().count())
          , min_time_(200000)
          , num_iters_for_timing_(num_iters_for_timing)
        {
        }

        /// Construct an \a persistent_auto_chunk_size executor parameters object
        ///
        /// \param rel_time     [in] The time duration to use as the minimum
        ///                     to decide how many loop iterations should be
        ///                     combined.
        /// \param time_cs       The execution time for each chunk.
        ///
        persistent_auto_chunk_size(pika::chrono::steady_duration const& time_cs,
            pika::chrono::steady_duration const& rel_time,
            std::uint64_t num_iters_for_timing = 0)
          : chunk_size_time_(time_cs.value().count())
          , min_time_(rel_time.value().count())
          , num_iters_for_timing_(num_iters_for_timing)
        {
        }

        /// \cond NOINTERNAL
        // Estimate a chunk size based on number of cores used.
        template <typename Executor, typename F>
        std::size_t get_chunk_size(
            Executor& /* exec */, F&& f, std::size_t cores, std::size_t count)
        {
            // by default use 1% of the iterations
            if (num_iters_for_timing_ == 0)
            {
                num_iters_for_timing_ = count / 100;
            }

            // perform measurements only if necessary
            if (num_iters_for_timing_ > 0)
            {
                using namespace std::chrono;

                auto t = steady_clock::now();

                std::size_t test_chunk_size = f(num_iters_for_timing_);
                if (test_chunk_size != 0)
                {
                    duration<unsigned long, std::nano> dur;
                    auto zero_duration = duration<unsigned long, std::nano>(0);
                    if (chunk_size_time_ == zero_duration)
                    {
                        dur = (steady_clock::now() - t) / test_chunk_size;
                        chunk_size_time_ = dur;
                    }
                    else
                    {
                        dur = chunk_size_time_;
                    }

                    if (dur != zero_duration && min_time_ >= dur)
                    {
                        // return chunk size which will create the required
                        // amount of work
                        return (std::min)(
                            count, (std::size_t)(min_time_ / dur));
                    }
                }
            }

            return (count + cores - 1) / cores;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::chrono::duration<unsigned long, std::nano> chunk_size_time_;
        std::chrono::duration<unsigned long, std::nano> min_time_;
        // number of iteration to use for timing
        std::uint64_t num_iters_for_timing_;
        /// \endcond
    };
}}    // namespace pika::execution

namespace pika { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<pika::execution::persistent_auto_chunk_size>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace pika::parallel::execution
