//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace pika::detail {
    template <typename T>
    class irange
    {
        static_assert(std::is_integral_v<T>);

    public:
        constexpr irange(T const b, T const e)
          : b(b)
          , e(e)
        {
            PIKA_ASSERT(b <= e);
        }
        constexpr irange(irange&&) = default;
        constexpr irange(irange const&) = default;
        constexpr irange& operator=(irange&&) = default;
        constexpr irange& operator=(irange const&) = default;

        class iterator
        {
        public:
            constexpr iterator() = default;
            constexpr explicit iterator(T const v) noexcept
              : value(v)
            {
            }
            constexpr iterator(iterator&&) = default;
            constexpr iterator(iterator const&) = default;
            constexpr iterator& operator=(iterator&&) = default;
            constexpr iterator& operator=(iterator const&) = default;

            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using reference = T&;
            using pointer = T*;
            using iterator_category = std::random_access_iterator_tag;

            constexpr reference operator*() noexcept
            {
                return value;
            }

            constexpr iterator& operator++() noexcept
            {
                ++value;
                return *this;
            }

            constexpr iterator operator++(int) noexcept
            {
                iterator r = *this;
                ++value;
                return r;
            }

            constexpr iterator& operator--() noexcept
            {
                --value;
                return *this;
            }

            constexpr iterator operator--(int) noexcept
            {
                iterator r = *this;
                --value;
                return r;
            }

            constexpr iterator& operator+=(difference_type const n) noexcept
            {
                value += n;
                return *this;
            }

            constexpr friend iterator operator+(
                iterator const it, difference_type const n) noexcept
            {
                return iterator{it.value + n};
            }

            constexpr friend iterator operator+(
                difference_type const n, iterator const it) noexcept
            {
                return iterator{it.value + n};
            }

            constexpr friend iterator operator-(
                iterator const it, difference_type const n) noexcept
            {
                return iterator{it.value - n};
            }

            constexpr iterator& operator-=(difference_type const n) noexcept
            {
                value -= n;
                return *this;
            }

            constexpr friend difference_type operator-(
                iterator const lhs, iterator const rhs) noexcept
            {
                return static_cast<difference_type>(lhs.value) -
                    static_cast<difference_type>(rhs.value);
            }

            constexpr reference operator[](difference_type d) const noexcept
            {
                return static_cast<value_type>(value + d);
            }

            constexpr friend bool operator==(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value == rhs.value;
            }

            constexpr friend bool operator!=(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value != rhs.value;
            }

            constexpr friend bool operator<(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value < rhs.value;
            }

            constexpr friend bool operator>(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value > rhs.value;
            }

            constexpr friend bool operator<=(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value <= rhs.value;
            }

            constexpr friend bool operator>=(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value >= rhs.value;
            }

        private:
            T value = 0;
        };

        constexpr iterator begin() const noexcept
        {
            return iterator{b};
        }

        constexpr iterator end() const noexcept
        {
            return iterator{e};
        }

    private:
        T b;
        T e;
    };

    template <typename T, typename U = T>
    class strided_irange
    {
        static_assert(std::is_integral_v<T>);

    public:
        constexpr strided_irange(
            T const b, T const e, T const step = 1) noexcept
          : b(b)
          , e(e)
          , step(step)
        {
            PIKA_ASSERT(step != 0);
            PIKA_ASSERT((b <= e && step > 0) || (e <= b && step < 0));
        }
        strided_irange(strided_irange&&) = default;
        strided_irange(strided_irange const&) = default;
        strided_irange& operator=(strided_irange&&) = default;
        strided_irange& operator=(strided_irange const&) = default;

        class iterator
        {
        public:
            constexpr iterator() = default;
            constexpr iterator(T const v, U const step) noexcept
              : value(v)
              , step(step)
            {
            }
            iterator(iterator&&) = default;
            iterator(iterator const&) = default;
            iterator& operator=(iterator&&) = default;
            iterator& operator=(iterator const&) = default;

            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using reference = T&;
            using pointer = T*;
            using iterator_category = std::random_access_iterator_tag;

            constexpr reference operator*() noexcept
            {
                return value;
            }

            constexpr iterator& operator++() noexcept
            {
                value += step;
                return *this;
            }

            constexpr iterator operator++(int) noexcept
            {
                iterator r = *this;
                value += step;
                return r;
            }

            constexpr iterator& operator--() noexcept
            {
                value -= step;
                return *this;
            }

            constexpr iterator operator--(int) noexcept
            {
                iterator r = *this;
                value -= step;
                return r;
            }

            constexpr iterator& operator+=(difference_type const n) noexcept
            {
                value += n * step;
                return *this;
            }

            constexpr friend iterator operator+(
                iterator const it, difference_type const n) noexcept
            {
                return iterator{it.value + n * it.step, it.step};
            }

            constexpr friend iterator operator+(
                difference_type const n, iterator const it) noexcept
            {
                return iterator{it.value + n * it.step, it.step};
            }

            constexpr friend iterator operator-(
                iterator const it, difference_type const n) noexcept
            {
                return iterator{it.value - n * it.step, it.step};
            }

            constexpr iterator& operator-=(difference_type const n) noexcept
            {
                value -= n * step;
                return *this;
            }

            constexpr friend difference_type operator-(
                iterator const lhs, iterator const rhs) noexcept
            {
                PIKA_ASSERT(lhs.step == rhs.step);
                return (static_cast<difference_type>(lhs.value) -
                           static_cast<difference_type>(rhs.value)) /
                    lhs.step;
            }

            constexpr reference operator[](difference_type d) const noexcept
            {
                return static_cast<value_type>(value + d * step);
            }

            constexpr friend bool operator==(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value == rhs.value;
            }

            constexpr friend bool operator!=(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value != rhs.value;
            }

            constexpr friend bool operator<(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value < rhs.value;
            }

            constexpr friend bool operator>(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value > rhs.value;
            }

            constexpr friend bool operator<=(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value <= rhs.value;
            }

            constexpr friend bool operator>=(
                iterator const lhs, iterator const rhs) noexcept
            {
                return lhs.value >= rhs.value;
            }

        private:
            T value = 0;
            U step = 0;
        };

        constexpr iterator begin() const noexcept
        {
            return iterator{b, step};
        }

        constexpr iterator end() const noexcept
        {
            if (b < e)
            {
                return iterator{b + ((e - b - 1) / step + 1) * step, step};
            }
            else if (b > e)
            {
                return iterator{b + ((e - b + 1) / step + 1) * step, step};
            }
            else
            {
                return iterator{b, step};
            }
        }

    private:
        T b;
        T e;
        U step;
    };

    template <typename T>
    irange(T const, T const) -> irange<std::decay_t<T>>;

    template <typename T, typename U = T>
    strided_irange(T const, T const, U const)
        -> strided_irange<std::decay_t<T>, std::decay_t<U>>;
}    // namespace pika::detail
