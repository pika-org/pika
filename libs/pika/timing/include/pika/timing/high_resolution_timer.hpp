//  Copyright (c) 2005-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <chrono>
#include <cstdint>

namespace pika::chrono::detail {
    ///////////////////////////////////////////////////////////////////////////
    //
    //  high_resolution_timer
    //      A timer object measures elapsed time.
    //
    ///////////////////////////////////////////////////////////////////////////
    class high_resolution_timer
    {
    public:
        high_resolution_timer() noexcept
          : start_time_(take_time_stamp())
        {
        }

        void restart() noexcept { start_time_ = take_time_stamp(); }

        // return elapsed time in seconds (double different from
        // std::chrono::seconds which returns an unsigned long)
        template <typename Unit = std::chrono::duration<double>>
        auto elapsed() const noexcept
        {
            return std::chrono::duration_cast<Unit>(take_time_stamp() - start_time_).count();
        }

        // return estimated maximum value for elapsed()
        static constexpr double elapsed_max() noexcept
        {
            return (std::chrono::duration_values<std::chrono::nanoseconds>::max)().count() * 1e-9;
        }

        // return minimum value for elapsed()
        static constexpr double elapsed_min() noexcept
        {
            return (std::chrono::duration_values<std::chrono::nanoseconds>::min)().count() * 1e-9;
        }

    protected:
        static std::chrono::time_point<std::chrono::high_resolution_clock>
        take_time_stamp() noexcept
        {
            return std::chrono::high_resolution_clock::now();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    };
}    // namespace pika::chrono::detail
