//  Copyright (c) 2020 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/concurrency/cache_line_data.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <atomic>
#include <cstdint>
#include <optional>
#include <type_traits>
#endif

namespace pika::concurrency::detail {
    /// \brief A concurrent queue which can only hold contiguous ranges of
    ///        integers.
    ///
    /// A concurrent queue which can be initialized with a range of integers.
    /// Items can be popped from both ends of the queue. Popping from the right
    /// decrements the next element that will be popped from the right, if there
    /// are items left. Popping from the left increments the next element that
    /// will be popped from the left, if there are items left.
    template <typename T = std::uint32_t>
    class contiguous_index_queue
    {
        static_assert(sizeof(T) <= 4,
            "contiguous_index_queue assumes at most 32 bit indices to fit two indices in an at "
            "most 64 bit struct");
        static_assert(
            std::is_integral_v<T>, "contiguous_index_queue only works with integral indices");

        struct range
        {
            T first = 0;
            T last = 0;

            range() = default;

            range(T first, T last) noexcept
              : first(first)
              , last(last)
            {
            }

            constexpr range increment_first() noexcept { return range{first + 1, last}; }

            constexpr range decrement_last() noexcept { return range{first, last - 1}; }

            constexpr bool empty() noexcept { return first >= last; }
        };

    public:
        /// \brief Reset the queue with the given range.
        ///
        /// Reset the queue with the given range. No additional synchronization
        /// is done to ensure that other threads are not accessing elements from
        /// the queue. It is the callees responsibility to ensure that it is
        /// safe to reset the queue.
        ///
        /// \param first Beginning of the new range.
        constexpr void reset(T first, T last) noexcept
        {
            initial_range = {first, last};
            current_range.data_ = {first, last};
            PIKA_ASSERT(first <= last);
        }

        /// \brief Construct a new contiguous_index_queue.
        ///
        /// Construct a new queue with an empty range.
        constexpr contiguous_index_queue() noexcept
          : initial_range{}
          , current_range{}
        {
        }

        /// \brief Construct a new contiguous_index_queue with the given range.
        ///
        /// Construct a new queue with the given range as the initial range.
        constexpr contiguous_index_queue(T first, T last) noexcept
          : initial_range{}
          , current_range{}
        {
            reset(first, last);
        }

        /// \brief Copy-construct a queue.
        ///
        /// No additional synchronization is done to ensure that other threads
        /// are not accessing elements from the queue being copied. It is the
        /// callees responsibility to ensure that it is safe to copy the queue.
        constexpr contiguous_index_queue(contiguous_index_queue<T> const& other)
          : initial_range{other.initial_range}
          , current_range{}
        {
            current_range.data_ = other.current_range.data_.load(std::memory_order_relaxed);
        }

        /// \brief Copy-assign a queue.
        ///
        /// No additional synchronization is done to ensure that other threads
        /// are not accessing elements from the queue being copied. It is the
        /// callees responsibility to ensure that it is safe to copy the queue.
        constexpr contiguous_index_queue& operator=(contiguous_index_queue const& other)
        {
            initial_range = other.initial_range;
            current_range = other.current_range.data_.load(std::memory_order_relaxed);
            return *this;
        }

        /// \brief Attempt to pop an item from the left of the queue.
        ///
        /// Attempt to pop an item from the left (beginning) of the queue. If
        /// no items are left std::nullopt is returned.
        constexpr std::optional<T> pop_left() noexcept
        {
            range desired_range{0, 0};
            T index = 0;

            range expected_range = current_range.data_.load(std::memory_order_relaxed);

            do {
                if (expected_range.empty()) { return std::nullopt; }

                index = expected_range.first;
                desired_range = expected_range.increment_first();
            } while (!current_range.data_.compare_exchange_weak(expected_range, desired_range));

            return std::make_optional<>(index);
        }

        /// \brief Attempt to pop an item from the right of the queue.
        ///
        /// Attempt to pop an item from the right (end) of the queue. If
        /// no items are left std::nullopt is returned.
        constexpr std::optional<T> pop_right() noexcept
        {
            range desired_range{0, 0};
            T index = 0;

            range expected_range = current_range.data_.load(std::memory_order_relaxed);

            do {
                if (expected_range.empty()) { return std::nullopt; }

                desired_range = expected_range.decrement_last();
                index = desired_range.last;
            } while (!current_range.data_.compare_exchange_weak(expected_range, desired_range));

            return std::make_optional(index);
        }

        constexpr bool empty() noexcept
        {
            return current_range.data_.load(std::memory_order_relaxed).empty();
        }

    private:
        range initial_range;
        cache_line_data<std::atomic<range>> current_range;
    };
}    // namespace pika::concurrency::detail
