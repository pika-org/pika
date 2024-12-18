//  Copyright (c) 2019 Austin McCartney
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/iterator_support/iterator_adaptor.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/testing.hpp>

#include <cstddef>
#include <forward_list>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

namespace test {
    template <typename BaseIterator, typename IteratorTag>
    struct test_iterator
      : pika::util::iterator_adaptor<test_iterator<BaseIterator, IteratorTag>, BaseIterator, void,
            IteratorTag>
    {
    private:
        using base_type = pika::util::iterator_adaptor<test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, void, IteratorTag>;

    public:
        test_iterator()
          : base_type()
        {
        }
        explicit test_iterator(BaseIterator base)
          : base_type(base)
        {
        }
    };
}    // namespace test

template <typename T, typename = void>
struct has_nested_type : std::integral_constant<bool, false>
{
};

template <typename T>
struct has_nested_type<T, std::void_t<typename T::type>> : std::integral_constant<bool, true>
{
};

struct bidirectional_traversal_iterator
{
    using difference_type = int;
    using value_type = int;
    using iterator_category = std::input_iterator_tag;
    using pointer = int const*;
    using reference = void;

    int state;

    int operator*() const { return this->state; }
    int operator->() const = delete;

    bidirectional_traversal_iterator& operator++()
    {
        ++(this->state);
        return *this;
    }

    bidirectional_traversal_iterator operator++(int)
    {
        bidirectional_traversal_iterator copy = *this;
        ++(*this);
        return copy;
    }

    bidirectional_traversal_iterator& operator--()
    {
        --(this->state);
        return *this;
    }

    bidirectional_traversal_iterator operator--(int)
    {
        bidirectional_traversal_iterator copy = *this;
        --(*this);
        return copy;
    }

    bool operator==(bidirectional_traversal_iterator const& that) const
    {
        return this->state == that.state;
    }

    bool operator!=(bidirectional_traversal_iterator const& that) const
    {
        return this->state != that.state;
    }

    bool operator<(bidirectional_traversal_iterator const& that) const
    {
        return this->state < that.state;
    }

    bool operator<=(bidirectional_traversal_iterator const& that) const
    {
        return this->state <= that.state;
    }

    bool operator>(bidirectional_traversal_iterator const& that) const
    {
        return this->state > that.state;
    }

    bool operator>=(bidirectional_traversal_iterator const& that) const
    {
        return this->state >= that.state;
    }
};

struct random_access_traversal_iterator
{
    using difference_type = int;
    using value_type = int;
    using iterator_category = std::input_iterator_tag;
    using pointer = int const*;
    using reference = void;

    int state;

    int operator*() const { return this->state; }
    int operator->() const = delete;

    random_access_traversal_iterator& operator++()
    {
        ++(this->state);
        return *this;
    }

    random_access_traversal_iterator operator++(int)
    {
        random_access_traversal_iterator copy = *this;
        ++(*this);
        return copy;
    }

    random_access_traversal_iterator& operator--()
    {
        --(this->state);
        return *this;
    }

    random_access_traversal_iterator operator--(int)
    {
        random_access_traversal_iterator copy = *this;
        --(*this);
        return copy;
    }

    int operator[](difference_type n) const { return this->state + n; }

    random_access_traversal_iterator& operator+=(difference_type n)
    {
        this->state += n;
        return *this;
    }

    random_access_traversal_iterator operator+(difference_type n)
    {
        random_access_traversal_iterator copy = *this;
        return copy += n;
    }

    random_access_traversal_iterator& operator-=(difference_type n)
    {
        this->state -= n;
        return *this;
    }

    random_access_traversal_iterator operator-(difference_type n)
    {
        random_access_traversal_iterator copy = *this;
        return copy -= n;
    }

    difference_type operator-(random_access_traversal_iterator const& that) const
    {
        return this->state - that.state;
    }

    bool operator==(random_access_traversal_iterator const& that) const
    {
        return this->state == that.state;
    }

    bool operator!=(random_access_traversal_iterator const& that) const
    {
        return this->state != that.state;
    }

    bool operator<(random_access_traversal_iterator const& that) const
    {
        return this->state < that.state;
    }

    bool operator<=(random_access_traversal_iterator const& that) const
    {
        return this->state <= that.state;
    }

    bool operator>(random_access_traversal_iterator const& that) const
    {
        return this->state > that.state;
    }

    bool operator>=(random_access_traversal_iterator const& that) const
    {
        return this->state >= that.state;
    }
};

void addition_result()
{
    using pika::traits::detail::addition_result;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator+(A const&) const { return B{}; }
    };

    PIKA_TEST_MSG((std::is_same<B, typename addition_result<C, A>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<addition_result<B, A>>::value), "invalid operation");
}

void dereference_result()
{
    using pika::traits::detail::dereference_result;

    struct A
    {
    };

    struct B
    {
        A operator*() const { return A{}; }
    };

    PIKA_TEST_MSG((std::is_same<A, typename dereference_result<B>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<dereference_result<A>>::value), "invalid operation");
}

void equality_result()
{
    using pika::detail::equality_result;
    using pika::detail::equality_result_t;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator==(A const&) const { return B{}; }
    };

    PIKA_TEST_MSG((std::is_same_v<B, equality_result_t<C, A>>), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<equality_result<B, A>>::value), "invalid operation");
}

void inequality_result()
{
    using pika::detail::inequality_result;
    using pika::detail::inequality_result_t;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator!=(A const&) const { return B{}; }
    };

    PIKA_TEST_MSG((std::is_same_v<B, inequality_result_t<C, A>>), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<inequality_result<B, A>>::value), "invalid operation");
}

void inplace_addition_result()
{
    using pika::traits::detail::inplace_addition_result;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator+=(A const&) const { return B{}; }
    };

    PIKA_TEST_MSG(
        (std::is_same<B, typename inplace_addition_result<C, A>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<inplace_addition_result<B, A>>::value), "invalid operation");
}

void inplace_subtraction_result()
{
    using pika::traits::detail::inplace_subtraction_result;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator-=(A const&) const { return B{}; }
    };

    PIKA_TEST_MSG(
        (std::is_same<B, typename inplace_subtraction_result<C, A>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<inplace_subtraction_result<B, A>>::value), "invalid operation");
}

void predecrement_result()
{
    using pika::traits::detail::predecrement_result;

    struct A
    {
    };

    struct B
    {
        A& operator--() const
        {
            static A a;
            return a;
        }
    };

    PIKA_TEST_MSG((std::is_same<A&, typename predecrement_result<B>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<predecrement_result<A>>::value), "invalid operation");
}

void preincrement_result()
{
    using pika::traits::detail::preincrement_result;

    struct A
    {
    };

    struct B
    {
        A& operator++() const
        {
            static A a;
            return a;
        }
    };

    PIKA_TEST_MSG((std::is_same<A&, typename preincrement_result<B>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<preincrement_result<A>>::value), "invalid operation");
}

void postdecrement_result()
{
    using pika::traits::detail::postdecrement_result;

    struct A
    {
    };

    struct B
    {
        A operator--(int) const { return A{}; }
    };

    PIKA_TEST_MSG((std::is_same<A, typename postdecrement_result<B>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<postdecrement_result<A>>::value), "invalid operation");
}

void postincrement_result()
{
    using pika::traits::detail::postincrement_result;

    struct A
    {
    };

    struct B
    {
        A operator++(int) const { return A{}; }
    };

    PIKA_TEST_MSG((std::is_same<A, typename postincrement_result<B>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<postincrement_result<A>>::value), "invalid operation");
}

void subscript_result()
{
    using pika::traits::detail::subscript_result;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator[](A const&) const { return B{}; }
    };

    PIKA_TEST_MSG((std::is_same<B, typename subscript_result<C, A>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<subscript_result<B, A>>::value), "invalid operation");
}

void subtraction_result()
{
    using pika::traits::detail::subtraction_result;

    struct A
    {
    };
    struct B
    {
    };

    struct C
    {
        B operator-(A const&) const { return B{}; }
    };

    PIKA_TEST_MSG(
        (std::is_same<B, typename subtraction_result<C, A>::type>::value), "deduced type");

    PIKA_TEST_MSG((!has_nested_type<subtraction_result<B, A>>::value), "invalid operation");
}

void bidirectional_concept()
{
    using pika::traits::detail::bidirectional_concept;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((!bidirectional_concept<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((!bidirectional_concept<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((!bidirectional_concept<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((bidirectional_concept<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((bidirectional_concept<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (bidirectional_concept<iterator>::value), "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (bidirectional_concept<iterator>::value), "random access traversal input iterator");
    }
}

void random_access_concept()
{
    using pika::traits::detail::addition_result;
    using pika::traits::detail::inplace_addition_result;
    using pika::traits::detail::inplace_subtraction_result;
    using pika::traits::detail::random_access_concept;
    using pika::traits::detail::subtraction_result;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((!random_access_concept<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((!random_access_concept<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((!random_access_concept<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((!random_access_concept<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((random_access_concept<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (!random_access_concept<iterator>::value), "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;

        using namespace pika::traits::detail;
        static_assert(
            std::is_same<iterator,
                typename addition_result<iterator,
                    typename std::iterator_traits<iterator>::difference_type>::type>::value,
            "");
        static_assert(
            std::is_same<typename std::add_lvalue_reference<iterator>::type,
                typename inplace_addition_result<iterator,
                    typename std::iterator_traits<iterator>::difference_type>::type>::value,
            "");
        static_assert(
            std::is_same<iterator,
                typename subtraction_result<iterator,
                    typename std::iterator_traits<iterator>::difference_type>::type>::value,
            "");
        static_assert(std::is_same<typename std::iterator_traits<iterator>::difference_type,
                          typename subtraction_result<iterator, iterator>::type>::value,
            "");
        static_assert(
            std::is_same<typename std::add_lvalue_reference<iterator>::type,
                typename inplace_subtraction_result<iterator,
                    typename std::iterator_traits<iterator>::difference_type>::type>::value,
            "");

        PIKA_TEST_MSG(
            (random_access_concept<iterator>::value), "random access traversal input iterator");
    }
}

void satisfy_traversal_concept_forward()
{
    using pika::traits::detail::satisfy_traversal_concept;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((!satisfy_traversal_concept<iterator, pika::forward_traversal_tag>::value),
            "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((!satisfy_traversal_concept<iterator, pika::forward_traversal_tag>::value),
            "input iterator");
    }
    {
        // see comment on definition for explanation
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((!satisfy_traversal_concept<iterator, pika::forward_traversal_tag>::value),
            "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((satisfy_traversal_concept<iterator, pika::forward_traversal_tag>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((satisfy_traversal_concept<iterator, pika::forward_traversal_tag>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG((satisfy_traversal_concept<iterator, pika::forward_traversal_tag>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG((satisfy_traversal_concept<iterator, pika::forward_traversal_tag>::value),
            "random access traversal input iterator");
    }
}

void satisfy_traversal_concept_bidirectional()
{
    using pika::traits::detail::satisfy_traversal_concept;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::bidirectional_traversal_tag>::value),
            "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::bidirectional_traversal_tag>::value),
            "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::bidirectional_traversal_tag>::value),
            "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG(
            (satisfy_traversal_concept<iterator, pika::bidirectional_traversal_tag>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG(
            (satisfy_traversal_concept<iterator, pika::bidirectional_traversal_tag>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (satisfy_traversal_concept<iterator, pika::bidirectional_traversal_tag>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (satisfy_traversal_concept<iterator, pika::bidirectional_traversal_tag>::value),
            "random access traversal input iterator");
    }
}

void satisfy_traversal_concept_random_access()
{
    using pika::traits::detail::satisfy_traversal_concept;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::random_access_traversal_tag>::value),
            "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::random_access_traversal_tag>::value),
            "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::random_access_traversal_tag>::value),
            "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::random_access_traversal_tag>::value),
            "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG(
            (satisfy_traversal_concept<iterator, pika::random_access_traversal_tag>::value),
            "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (!satisfy_traversal_concept<iterator, pika::random_access_traversal_tag>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (satisfy_traversal_concept<iterator, pika::random_access_traversal_tag>::value),
            "random access traversal input iterator");
    }
}

void is_iterator()
{
    using pika::traits::is_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((is_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((is_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((is_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((is_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((is_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG((is_iterator<iterator>::value), "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG((is_iterator<iterator>::value), "random access traversal input iterator");
    }
}

void is_output_iterator()
{
    using pika::traits::is_output_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((is_output_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((!is_output_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((is_output_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((is_output_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((is_output_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (is_output_iterator<iterator>::value), "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (is_output_iterator<iterator>::value), "random access traversal input iterator");
    }
}

void is_input_iterator()
{
    using pika::traits::is_input_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((!is_input_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (is_input_iterator<iterator>::value), "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (is_input_iterator<iterator>::value), "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::forward_iterator_tag>;

        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::input_iterator_tag>;

        PIKA_TEST_MSG((is_input_iterator<iterator>::value), "pika test iterator");
    }
}

void is_forward_iterator()
{
    using pika::traits::is_forward_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((!is_forward_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((!is_forward_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((is_forward_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((is_forward_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((is_forward_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (is_forward_iterator<iterator>::value), "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (is_forward_iterator<iterator>::value), "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        PIKA_TEST_MSG((is_forward_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        PIKA_TEST_MSG((is_forward_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::forward_iterator_tag>;

        PIKA_TEST_MSG((is_forward_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::input_iterator_tag>;

        PIKA_TEST_MSG((!is_forward_iterator<iterator>::value), "pika test iterator");
    }
}

void is_bidirectional_iterator()
{
    using pika::traits::is_bidirectional_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((!is_bidirectional_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((!is_bidirectional_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((!is_bidirectional_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((is_bidirectional_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((is_bidirectional_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG(
            (is_bidirectional_iterator<iterator>::value), "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (is_bidirectional_iterator<iterator>::value), "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        PIKA_TEST_MSG((is_bidirectional_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        PIKA_TEST_MSG((is_bidirectional_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::forward_iterator_tag>;

        PIKA_TEST_MSG((!is_bidirectional_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::input_iterator_tag>;

        PIKA_TEST_MSG((!is_bidirectional_iterator<iterator>::value), "pika test iterator");
    }
}

void is_random_access_iterator()
{
    using pika::traits::is_random_access_iterator;

    {
        using iterator = std::ostream_iterator<int>;
        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value), "output iterator");
    }
    {
        using iterator = std::istream_iterator<int>;
        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value), "input iterator");
    }
    {
        using iterator = typename std::forward_list<int>::iterator;
        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value), "forward iterator");
    }
    {
        using iterator = typename std::list<int>::iterator;
        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value), "bidirectional iterator");
    }
    {
        using iterator = typename std::vector<int>::iterator;
        PIKA_TEST_MSG((is_random_access_iterator<iterator>::value), "random access iterator");
    }
    {
        using iterator = bidirectional_traversal_iterator;
        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value),
            "bidirectional traversal input iterator");
    }
    {
        using iterator = random_access_traversal_iterator;
        PIKA_TEST_MSG(
            (is_random_access_iterator<iterator>::value), "random access traversal input iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::random_access_iterator_tag>;

        PIKA_TEST_MSG((is_random_access_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::bidirectional_iterator_tag>;

        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::forward_iterator_tag>;

        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value), "pika test iterator");
    }
    {
        using base_iterator = std::vector<std::size_t>::iterator;
        using iterator = test::test_iterator<base_iterator, std::input_iterator_tag>;

        PIKA_TEST_MSG((!is_random_access_iterator<iterator>::value), "pika test iterator");
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        addition_result();
        dereference_result();
        equality_result();
        inequality_result();
        inplace_addition_result();
        inplace_subtraction_result();
        predecrement_result();
        preincrement_result();
        postdecrement_result();
        postincrement_result();
        subscript_result();
        subtraction_result();
        bidirectional_concept();
        random_access_concept();
        satisfy_traversal_concept_forward();
        satisfy_traversal_concept_bidirectional();
        satisfy_traversal_concept_random_access();
        is_iterator();
        is_forward_iterator();
        is_bidirectional_iterator();
        is_random_access_iterator();
    }

    return 0;
}
