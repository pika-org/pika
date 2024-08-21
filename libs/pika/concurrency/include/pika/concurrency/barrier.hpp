//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#endif

#include <pika/config/warnings_prefix.hpp>

#if !defined(CHAR_BIT)
#define CHAR_BIT 8
#endif

namespace pika::concurrency::detail {
    class PIKA_EXPORT barrier
    {
    private:
        using mutex_type = std::mutex;

        static constexpr std::size_t barrier_flag = static_cast<std::size_t>(1)
            << (CHAR_BIT * sizeof(std::size_t) - 1);

    public:
        barrier(std::size_t number_of_threads);
        ~barrier();

        void wait();

    private:
        std::size_t const number_of_threads_;
        std::size_t total_;

        mutable mutex_type mtx_;
        std::condition_variable cond_;
    };
}    // namespace pika::concurrency::detail

#include <pika/config/warnings_suffix.hpp>
