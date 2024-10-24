//  (C) Copyright Thomas Witt 2003.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/iterator_adaptor.hpp>
#include <pika/iterator_support/tests/iterator_tests.hpp>
#include <pika/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <list>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

struct mult_functor
{
    // Functors used with transform_iterator must be
    // DefaultConstructible, as the transform_iterator must be
    // DefaultConstructible to satisfy the requirements for
    // TrivialIterator.
    mult_functor() {}
    mult_functor(int aa)
      : a(aa)
    {
    }

    int operator()(int b) const { return a * b; }

    int a;
};

template <typename Pair>
struct select1st_
{
    const typename Pair::first_type& operator()(Pair const& x) const { return x.first; }

    typename Pair::first_type& operator()(Pair& x) const { return x.first; }
};

struct one_or_four
{
    bool operator()(tests::dummy_type x) const { return x.foo() == 1 || x.foo() == 4; }
};

using storage = std::deque<int>;
using pointer_deque = std::deque<int*>;
using iterator_set = std::set<storage::iterator>;

template <class T>
struct foo;

void blah(int) {}

struct my_gen
{
    using result_type = int;

    my_gen()
      : n(0)
    {
    }

    int operator()() { return ++n; }

    int n;
};

template <typename V>
struct ptr_iterator
  : pika::util::iterator_adaptor<ptr_iterator<V>, V*, V, std::random_access_iterator_tag>
{
private:
    using base_adaptor_type =
        pika::util::iterator_adaptor<ptr_iterator<V>, V*, V, std::random_access_iterator_tag>;

public:
    ptr_iterator() {}

    ptr_iterator(V* d)
      : base_adaptor_type(d)
    {
    }

    template <typename V2>
    ptr_iterator(
        ptr_iterator<V2> const& x, std::enable_if_t<std::is_convertible<V2*, V*>::value>* = nullptr)
      : base_adaptor_type(x.base())
    {
    }
};

// Non-functional iterator for category modification checking
template <typename Iter, typename Category>
struct modify_traversal
  : pika::util::iterator_adaptor<modify_traversal<Iter, Category>, Iter, void, Category>
{
};

template <typename T>
struct fwd_iterator
  : pika::util::iterator_adaptor<fwd_iterator<T>, tests::forward_iterator_archetype<T>>
{
private:
    using base_adaptor_type =
        pika::util::iterator_adaptor<fwd_iterator<T>, tests::forward_iterator_archetype<T>>;

public:
    fwd_iterator() {}

    fwd_iterator(tests::forward_iterator_archetype<T> d)
      : base_adaptor_type(d)
    {
    }
};

template <typename T>
struct in_iterator
  : pika::util::iterator_adaptor<in_iterator<T>, tests::input_iterator_archetype_no_proxy<T>>
{
private:
    using base_adaptor_type =
        pika::util::iterator_adaptor<in_iterator<T>, tests::input_iterator_archetype_no_proxy<T>>;

public:
    in_iterator() {}
    in_iterator(tests::input_iterator_archetype_no_proxy<T> d)
      : base_adaptor_type(d)
    {
    }
};

template <typename Iter>
struct constant_iterator
  : pika::util::iterator_adaptor<constant_iterator<Iter>, Iter,
        typename std::iterator_traits<Iter>::value_type const>
{
    using base_adaptor_type = pika::util::iterator_adaptor<constant_iterator<Iter>, Iter,
        const typename std::iterator_traits<Iter>::value_type>;

    constant_iterator() {}
    constant_iterator(Iter it)
      : base_adaptor_type(it)
    {
    }
};

int main()
{
    tests::dummy_type array[] = {tests::dummy_type(0), tests::dummy_type(1), tests::dummy_type(2),
        tests::dummy_type(3), tests::dummy_type(4), tests::dummy_type(5)};
    int const N = sizeof(array) / sizeof(tests::dummy_type);

    // sanity check, if this doesn't pass the test is buggy
    tests::random_access_iterator_test(array, N, array);

    // Test the iterator_adaptor
    {
        ptr_iterator<tests::dummy_type> i(array);
        tests::random_access_iterator_test(i, N, array);

        ptr_iterator<const tests::dummy_type> j(array);
        tests::random_access_iterator_test(j, N, array);
        tests::const_nonconst_iterator_test(i, ++j);
    }

    // Test the iterator_traits
    {
        // Test computation of defaults
        using Iter1 = ptr_iterator<int>;

        // don't use std::iterator_traits here to avoid VC++ problems
        PIKA_TEST((std::is_same<Iter1::value_type, int>::value));
        PIKA_TEST((std::is_same<Iter1::reference, int&>::value));
        PIKA_TEST((std::is_same<Iter1::pointer, int*>::value));
        PIKA_TEST((std::is_same<Iter1::difference_type, std::ptrdiff_t>::value));

        PIKA_TEST((
            std::is_convertible<Iter1::iterator_category, std::random_access_iterator_tag>::value));
    }

    {
        // Test computation of default when the Value is const
        using Iter1 = ptr_iterator<int const>;
        PIKA_TEST((std::is_same<Iter1::value_type, int>::value));
        PIKA_TEST((std::is_same<Iter1::reference, int const&>::value));

        //PIKA_TEST(boost::is_readable_iterator<Iter1>::value);
        //PIKA_TEST(boost::is_lvalue_iterator<Iter1>::value);

        PIKA_TEST((std::is_same<Iter1::pointer, int const*>::value));
    }

    {
        // Test constant iterator idiom
        using BaseIter = ptr_iterator<int>;
        using Iter = constant_iterator<BaseIter>;

        Iter it;

        PIKA_TEST((std::is_same<Iter::value_type, int>::value));
        PIKA_TEST((std::is_same<Iter::reference, int const&>::value));
        PIKA_TEST((std::is_same<Iter::pointer, int const*>::value));

        //PIKA_TEST(boost::is_non_const_lvalue_iterator<BaseIter>::value);
        //PIKA_TEST(boost::is_lvalue_iterator<Iter>::value);

        using IncrementableIter = modify_traversal<BaseIter, std::input_iterator_tag>;

        PIKA_TEST(
            (std::is_same<BaseIter::iterator_category, std::random_access_iterator_tag>::value));
        PIKA_TEST(
            (std::is_same<IncrementableIter::iterator_category, std::input_iterator_tag>::value));
    }

    // Test the iterator_adaptor
    {
        ptr_iterator<tests::dummy_type> i(array);
        tests::random_access_iterator_test(i, N, array);

        ptr_iterator<const tests::dummy_type> j(array);
        tests::random_access_iterator_test(j, N, array);
        tests::const_nonconst_iterator_test(i, ++j);
    }

    // check operator-> with a forward iterator
    {
        tests::forward_iterator_archetype<tests::dummy_type> forward_iter;

        using adaptor_type = fwd_iterator<tests::dummy_type>;

        adaptor_type i(forward_iter);
        int zero = 0;
        if (zero)    // don't do this, just make sure it compiles
        {
            PIKA_TEST_EQ((*i).x_, i->foo());
        }
    }

    // check operator-> with an input iterator
    {
        tests::input_iterator_archetype_no_proxy<tests::dummy_type> input_iter;
        using adaptor_type = in_iterator<tests::dummy_type>;
        adaptor_type i(input_iter);
        int zero = 0;
        if (zero)    // don't do this, just make sure it compiles
        {
            PIKA_TEST_EQ((*i).x_, i->foo());
        }
    }

    // check that base_type is correct
    {
        // Test constant iterator idiom
        using BaseIter = ptr_iterator<int>;

        PIKA_TEST((std::is_same<BaseIter::base_type, int*>::value));
        PIKA_TEST((std::is_same<constant_iterator<BaseIter>::base_type, BaseIter>::value));

        using IncrementableIter = modify_traversal<BaseIter, std::forward_iterator_tag>;

        PIKA_TEST((std::is_same<IncrementableIter::base_type, BaseIter>::value));
    }

    return 0;
}
