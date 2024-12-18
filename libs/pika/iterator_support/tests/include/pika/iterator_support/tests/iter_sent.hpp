//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2019 Hartmut Kaiser
//  Copyright (c) 2019 Piotr Mikolajczyk
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/iterator_support/traits/is_iterator.hpp>

#include <cstddef>
#include <iterator>

template <typename ValueType>
struct sentinel
{
    explicit sentinel(ValueType stop_value)
      : stop(stop_value)
    {
    }

    ValueType get_stop() const { return this->stop; }

private:
    ValueType stop;
};

template <typename Iter, typename ValueType,
    typename Enable = std::enable_if_t<pika::traits::is_forward_iterator<Iter>::value>>
bool operator==(Iter it, sentinel<ValueType> s)
{
    return *it == s.get_stop();
}

template <typename Iter, typename ValueType,
    typename Enable = std::enable_if_t<pika::traits::is_forward_iterator<Iter>::value>>
bool operator==(sentinel<ValueType> s, Iter it)
{
    return *it == s.get_stop();
}

template <typename Iter, typename ValueType,
    typename Enable = std::enable_if_t<pika::traits::is_forward_iterator<Iter>::value>>
bool operator!=(Iter it, sentinel<ValueType> s)
{
    return *it != s.get_stop();
}

template <typename Iter, typename ValueType,
    typename Enable = std::enable_if_t<pika::traits::is_forward_iterator<Iter>::value>>
bool operator!=(sentinel<ValueType> s, Iter it)
{
    return *it != s.get_stop();
}

template <typename Value>
struct iterator
{
    using difference_type = std::ptrdiff_t;
    using value_type = Value;
    using iterator_category = std::forward_iterator_tag;
    using pointer = Value const*;
    using reference = Value const&;

    explicit iterator(Value initialState)
      : state(initialState)
    {
    }

    virtual Value operator*() const { return this->state; }

    virtual Value operator->() const = delete;

    iterator& operator++()
    {
        ++(this->state);
        return *this;
    }

    iterator operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    iterator& operator--()
    {
        --(this->state);
        return *this;
    }

    iterator operator--(int)
    {
        auto copy = *this;
        --(*this);
        return copy;
    }

    virtual Value operator[](difference_type n) const { return this->state + n; }

    iterator& operator+=(difference_type n)
    {
        this->state += n;
        return *this;
    }

    iterator operator+(difference_type n) const
    {
        iterator copy = *this;
        return copy += n;
    }

    iterator& operator-=(difference_type n)
    {
        this->state -= n;
        return *this;
    }

    iterator operator-(difference_type n) const
    {
        iterator copy = *this;
        return copy -= n;
    }

    bool operator==(iterator const& that) const { return this->state == that.state; }

    bool operator!=(iterator const& that) const { return this->state != that.state; }

    bool operator<(iterator const& that) const { return this->state < that.state; }

    bool operator<=(iterator const& that) const { return this->state <= that.state; }

    bool operator>(iterator const& that) const { return this->state > that.state; }

    bool operator>=(iterator const& that) const { return this->state >= that.state; }

protected:
    Value state;
};
