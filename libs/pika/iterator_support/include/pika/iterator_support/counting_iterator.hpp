//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code is based on boost::iterators::counting_iterator
//  Copyright David Abrahams 2003.

#pragma once

#include <pika/config.hpp>
#include <pika/iterator_support/iterator_adaptor.hpp>
#include <pika/iterator_support/iterator_range.hpp>
#include <pika/iterator_support/range.hpp>
#include <pika/iterator_support/traits/is_range.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/type_support/lazy_conditional.hpp>
#include <pika/type_support/type_identity.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#endif

namespace pika::util {

    template <typename Incrementable, typename CategoryOrTraversal = void,
        typename Difference = void, typename Enable = void>
    class counting_iterator;

    namespace detail {

        template <typename Incrementable, typename CategoryOrTraversal, typename Difference>
        struct counting_iterator_base
        {
            // calculate category of the resulting iterator
            template <typename Iterator>
            struct iterator_category
            {
                using type = typename std::iterator_traits<Iterator>::iterator_category;
            };

            using base_traversal =
                ::pika::detail::lazy_conditional<std::is_integral<Incrementable>::value,
                    ::pika::detail::type_identity<std::random_access_iterator_tag>,
                    iterator_category<Incrementable>>;

            using traversal =
                ::pika::detail::lazy_conditional_t<std::is_void<CategoryOrTraversal>::value,
                    base_traversal, ::pika::detail::type_identity<CategoryOrTraversal>>;

            // calculate difference_type of the resulting iterator
            template <typename Integer>
            struct integer_difference_type
            {
                using type = typename std::conditional<(sizeof(Integer) >= sizeof(std::intmax_t)),

                    std::conditional<(std::is_signed<Integer>::value), Integer, std::intmax_t>,

                    std::conditional<(sizeof(Integer) < sizeof(std::ptrdiff_t)), std::ptrdiff_t,
                        std::intmax_t>>::type::type;
            };

            template <typename Iterator>
            struct iterator_difference_type
            {
                using type = typename std::iterator_traits<Iterator>::difference_type;
            };

            using base_difference =
                ::pika::detail::lazy_conditional<std::is_integral<Incrementable>::value,
                    integer_difference_type<Incrementable>,
                    iterator_difference_type<Incrementable>>;

            using difference = ::pika::detail::lazy_conditional_t<std::is_void<Difference>::value,
                base_difference, ::pika::detail::type_identity<Difference>>;

            using type =
                iterator_adaptor<counting_iterator<Incrementable, CategoryOrTraversal, Difference>,
                    Incrementable, Incrementable, traversal, Incrementable const&, difference>;
        };
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    // specialization for Iterators (non-integral types)
    template <typename Incrementable, typename CategoryOrTraversal, typename Difference,
        typename Enable>
    class counting_iterator
      : public detail::counting_iterator_base<Incrementable, CategoryOrTraversal, Difference>::type
    {
    private:
        using base_type = typename detail::counting_iterator_base<Incrementable,
            CategoryOrTraversal, Difference>::type;

        friend class iterator_core_access;

    public:
        counting_iterator() = default;
        counting_iterator(counting_iterator&& rhs) = default;
        counting_iterator& operator=(counting_iterator&& rhs) = default;
        counting_iterator(counting_iterator const& rhs) = default;
        counting_iterator& operator=(counting_iterator const& rhs) = default;

        PIKA_HOST_DEVICE explicit counting_iterator(Incrementable x)
          : base_type(x)
        {
        }

    private:
        PIKA_HOST_DEVICE typename base_type::reference dereference() const
        {
            return this->base_reference();
        }
    };

    template <typename Incrementable, typename CategoryOrTraversal, typename Difference>
    class counting_iterator<Incrementable, CategoryOrTraversal, Difference,
        std::enable_if_t<std::is_integral<Incrementable>::value>>
      : public detail::counting_iterator_base<Incrementable, CategoryOrTraversal, Difference>::type
    {
    private:
        using base_type = typename detail::counting_iterator_base<Incrementable,
            CategoryOrTraversal, Difference>::type;

        friend class iterator_core_access;

    public:
        counting_iterator() = default;
        counting_iterator(counting_iterator&& rhs) = default;
        counting_iterator& operator=(counting_iterator&& rhs) = default;
        counting_iterator(counting_iterator const& rhs) = default;
        counting_iterator& operator=(counting_iterator const& rhs) = default;

        PIKA_HOST_DEVICE explicit counting_iterator(Incrementable x)
          : base_type(x)
        {
        }

    private:
        template <typename Iterator>
        PIKA_HOST_DEVICE bool equal(Iterator const& rhs) const
        {
            return this->base() == rhs.base();
        }

        PIKA_HOST_DEVICE void increment() { ++this->base_reference(); }

        PIKA_HOST_DEVICE void decrement() { --this->base_reference(); }

        template <typename Distance>
        PIKA_HOST_DEVICE void advance(Distance n)
        {
            this->base_reference() += static_cast<typename base_type::base_type>(n);
        }

        PIKA_HOST_DEVICE typename base_type::reference dereference() const
        {
            return this->base_reference();
        }

        template <typename OtherIncrementable>
        PIKA_HOST_DEVICE typename base_type::difference_type distance_to(
            counting_iterator<OtherIncrementable, CategoryOrTraversal, Difference> const& y) const
        {
            using difference_type = typename base_type::difference_type;
            return static_cast<difference_type>(y.base()) -
                static_cast<difference_type>(this->base());
        }
    };

    // Manufacture a counting iterator for an arbitrary incrementable type
    template <typename Incrementable>
    PIKA_HOST_DEVICE inline counting_iterator<Incrementable> make_counting_iterator(Incrementable x)
    {
        return counting_iterator<Incrementable>(x);
    }
}    // namespace pika::util
