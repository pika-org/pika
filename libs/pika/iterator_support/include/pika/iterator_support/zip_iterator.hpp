//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/functional/invoke_result.hpp>
#include <pika/iterator_support/iterator_facade.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/serialization/serialization_fwd.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika { namespace util {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_value;

        template <typename... Ts>
        struct zip_iterator_value<pika::tuple<Ts...>>
        {
            using type =
                pika::tuple<typename std::iterator_traits<Ts>::value_type...>;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct zip_iterator_reference;

        template <typename... Ts>
        struct zip_iterator_reference<pika::tuple<Ts...>>
        {
            using type =
                pika::tuple<typename std::iterator_traits<Ts>::reference...>;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename U>
        struct zip_iterator_category_impl
        {
            static_assert(sizeof(T) == 0 && sizeof(U) == 0,
                "unknown combination of iterator categories");
        };

        // random_access_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::random_access_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::random_access_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::random_access_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // bidirectional_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::bidirectional_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::bidirectional_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::bidirectional_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // forward_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::forward_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::forward_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        template <>
        struct zip_iterator_category_impl<std::forward_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        // input_iterator_tag
        template <>
        struct zip_iterator_category_impl<std::input_iterator_tag,
            std::input_iterator_tag>
        {
            typedef std::input_iterator_tag type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Enable = void>
        struct zip_iterator_category;

        template <typename T>
        struct zip_iterator_category<pika::tuple<T>,
            typename std::enable_if<pika::tuple_size<pika::tuple<T>>::value ==
                1>::type>
        {
            typedef typename std::iterator_traits<T>::iterator_category type;
        };

        template <typename T, typename U>
        struct zip_iterator_category<pika::tuple<T, U>,
            typename std::enable_if<pika::tuple_size<pika::tuple<T, U>>::value ==
                2>::type>
          : zip_iterator_category_impl<
                typename std::iterator_traits<T>::iterator_category,
                typename std::iterator_traits<U>::iterator_category>
        {
        };

        template <typename T, typename U, typename... Tail>
        struct zip_iterator_category<pika::tuple<T, U, Tail...>,
            typename std::enable_if<(
                pika::tuple_size<pika::tuple<T, U, Tail...>>::value > 2)>::type>
          : zip_iterator_category_impl<
                typename zip_iterator_category_impl<
                    typename std::iterator_traits<T>::iterator_category,
                    typename std::iterator_traits<U>::iterator_category>::type,
                typename zip_iterator_category<pika::tuple<Tail...>>::type>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple>
        struct dereference_iterator;

        template <typename... Ts>
        struct dereference_iterator<pika::tuple<Ts...>>
        {
            template <std::size_t... Is>
            PIKA_HOST_DEVICE static
                typename zip_iterator_reference<pika::tuple<Ts...>>::type
                call(
                    util::index_pack<Is...>, pika::tuple<Ts...> const& iterators)
            {
                return pika::forward_as_tuple(*pika::get<Is>(iterators)...);
            }
        };

        struct increment_iterator
        {
            template <typename T>
            PIKA_HOST_DEVICE void operator()(T& iter) const
            {
                ++iter;
            }
        };

        struct decrement_iterator
        {
            template <typename T>
            PIKA_HOST_DEVICE void operator()(T& iter) const
            {
                --iter;
            }
        };

        struct advance_iterator
        {
            explicit advance_iterator(std::ptrdiff_t n)
              : n_(n)
            {
            }

            template <typename T>
            PIKA_HOST_DEVICE void operator()(T& iter) const
            {
                iter += n_;
            }

            std::ptrdiff_t n_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename IteratorTuple, typename Derived>
        class zip_iterator_base
          : public pika::util::iterator_facade<Derived,
                typename zip_iterator_value<IteratorTuple>::type,
                typename zip_iterator_category<IteratorTuple>::type,
                typename zip_iterator_reference<IteratorTuple>::type>
        {
            typedef pika::util::iterator_facade<
                zip_iterator_base<IteratorTuple, Derived>,
                typename zip_iterator_value<IteratorTuple>::type,
                typename zip_iterator_category<IteratorTuple>::type,
                typename zip_iterator_reference<IteratorTuple>::type>
                base_type;

        public:
            PIKA_HOST_DEVICE zip_iterator_base() {}

            PIKA_HOST_DEVICE
            zip_iterator_base(IteratorTuple const& iterators)
              : iterators_(iterators)
            {
            }
            PIKA_HOST_DEVICE
            zip_iterator_base(IteratorTuple&& iterators)
              : iterators_(PIKA_MOVE(iterators))
            {
            }

            typedef IteratorTuple iterator_tuple_type;

            PIKA_HOST_DEVICE iterator_tuple_type get_iterator_tuple() const
            {
                return iterators_;
            }

        private:
            friend class pika::util::iterator_core_access;

            PIKA_HOST_DEVICE bool equal(zip_iterator_base const& other) const
            {
                return iterators_ == other.iterators_;
            }

            PIKA_HOST_DEVICE typename base_type::reference dereference() const
            {
                return dereference_iterator<IteratorTuple>::call(
                    typename util::make_index_pack<
                        pika::tuple_size<IteratorTuple>::value>::type(),
                    iterators_);
            }

            PIKA_HOST_DEVICE void increment()
            {
                this->apply(increment_iterator());
            }

            PIKA_HOST_DEVICE void decrement()
            {
                this->apply(decrement_iterator());
            }

            PIKA_HOST_DEVICE void advance(std::ptrdiff_t n)
            {
                this->apply(advance_iterator(n));
            }

            PIKA_HOST_DEVICE
            std::ptrdiff_t distance_to(zip_iterator_base const& other) const
            {
                return pika::get<0>(other.iterators_) - pika::get<0>(iterators_);
            }

        private:
            template <typename F, std::size_t... Is>
            PIKA_HOST_DEVICE void apply(F&& f, util::index_pack<Is...>)
            {
                int const _sequencer[] = {
                    ((f(pika::get<Is>(iterators_))), 0)...};
                (void) _sequencer;
            }

            template <typename F>
            PIKA_HOST_DEVICE void apply(F&& f)
            {
                return apply(PIKA_FORWARD(F, f),
                    util::make_index_pack<
                        pika::tuple_size<IteratorTuple>::value>());
            }

        private:
            friend class pika::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned)
            {
                ar& iterators_;
            }

        private:
            IteratorTuple iterators_;
        };
    }    // namespace detail

    template <typename... Ts>
    class zip_iterator
      : public detail::zip_iterator_base<pika::tuple<Ts...>, zip_iterator<Ts...>>
    {
        static_assert(
            sizeof...(Ts) != 0, "zip_iterator must wrap at least one iterator");

        typedef detail::zip_iterator_base<pika::tuple<Ts...>,
            zip_iterator<Ts...>>
            base_type;

    public:
        PIKA_HOST_DEVICE zip_iterator()
          : base_type()
        {
        }

        PIKA_HOST_DEVICE explicit zip_iterator(Ts const&... vs)
          : base_type(pika::tie(vs...))
        {
        }

        PIKA_HOST_DEVICE explicit zip_iterator(pika::tuple<Ts...>&& t)
          : base_type(PIKA_MOVE(t))
        {
        }

        PIKA_HOST_DEVICE zip_iterator(zip_iterator const& other)
          : base_type(other)
        {
        }

        PIKA_HOST_DEVICE zip_iterator(zip_iterator&& other)
          : base_type(PIKA_MOVE(other))
        {
        }

        PIKA_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }
        PIKA_HOST_DEVICE zip_iterator& operator=(zip_iterator&& other)
        {
            base_type::operator=(PIKA_MOVE(other));
            return *this;
        }

        template <typename... Ts_>
        PIKA_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }
        template <typename... Ts_>
        PIKA_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...>&& other)
        {
            base_type::operator=(
                base_type(PIKA_MOVE(other.get_iterator_tuple())));
            return *this;
        }
    };

    template <typename... Ts>
    class zip_iterator<pika::tuple<Ts...>>
      : public detail::zip_iterator_base<pika::tuple<Ts...>,
            zip_iterator<pika::tuple<Ts...>>>
    {
        static_assert(
            sizeof...(Ts) != 0, "zip_iterator must wrap at least one iterator");

        using base_type = detail::zip_iterator_base<pika::tuple<Ts...>,
            zip_iterator<pika::tuple<Ts...>>>;

    public:
        PIKA_HOST_DEVICE zip_iterator()
          : base_type()
        {
        }

        PIKA_HOST_DEVICE explicit zip_iterator(Ts const&... vs)
          : base_type(pika::tie(vs...))
        {
        }

        PIKA_HOST_DEVICE explicit zip_iterator(pika::tuple<Ts...>&& t)
          : base_type(PIKA_MOVE(t))
        {
        }

        PIKA_HOST_DEVICE zip_iterator(zip_iterator const& other)
          : base_type(other)
        {
        }

        PIKA_HOST_DEVICE zip_iterator(zip_iterator&& other)
          : base_type(PIKA_MOVE(other))
        {
        }

        PIKA_HOST_DEVICE zip_iterator& operator=(zip_iterator const& other)
        {
            base_type::operator=(other);
            return *this;
        }
        PIKA_HOST_DEVICE zip_iterator& operator=(zip_iterator&& other)
        {
            base_type::operator=(PIKA_MOVE(other));
            return *this;
        }

        template <typename... Ts_>
        PIKA_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...> const& other)
        {
            base_type::operator=(base_type(other.get_iterator_tuple()));
            return *this;
        }
        template <typename... Ts_>
        PIKA_HOST_DEVICE typename std::enable_if<
            std::is_assignable<typename zip_iterator::iterator_tuple_type&,
                typename zip_iterator<Ts_...>::iterator_tuple_type&&>::value,
            zip_iterator&>::type
        operator=(zip_iterator<Ts_...>&& other)
        {
            base_type::operator=(
                base_type(PIKA_MOVE(other.get_iterator_tuple())));
            return *this;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    PIKA_HOST_DEVICE zip_iterator<typename std::decay<Ts>::type...>
    make_zip_iterator(Ts&&... vs)
    {
        typedef zip_iterator<typename std::decay<Ts>::type...> result_type;

        return result_type(PIKA_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    struct zip_iterator_category
      : detail::zip_iterator_category<typename ZipIter::iterator_tuple_type>
    {
    };
}}    // namespace pika::util

namespace pika { namespace traits {

    namespace functional {

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T>
        struct element_result_of : util::invoke_result<F, T>
        {
        };

        template <typename F, typename Iter>
        struct lift_zipped_iterators;

        template <typename F, typename... Ts>
        struct lift_zipped_iterators<F, util::zip_iterator<Ts...>>
        {
            typedef typename util::zip_iterator<Ts...>::iterator_tuple_type
                tuple_type;
            typedef pika::tuple<typename element_result_of<
                typename F::template apply<Ts>, Ts>::type...>
                result_type;

            template <std::size_t... Is, typename... Ts_>
            static result_type call(
                util::index_pack<Is...>, pika::tuple<Ts_...> const& t)
            {
                return pika::make_tuple(
                    typename F::template apply<Ts>()(pika::get<Is>(t))...);
            }

            template <typename... Ts_>
            static result_type call(util::zip_iterator<Ts_...> const& iter)
            {
                using pika::util::make_index_pack;
                return call(typename make_index_pack<sizeof...(Ts)>::type(),
                    iter.get_iterator_tuple());
            }
        };
    }    // namespace functional

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Iter>
    struct is_zip_iterator<pika::util::zip_iterator<Iter...>> : std::true_type
    {
    };
}}    // namespace pika::traits
