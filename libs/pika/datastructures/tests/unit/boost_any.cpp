// Copyright Kevlin Henney, 2000, 2001. All rights reserved.
// Copyright (c) 2013 Hartmut Kaiser.
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <pika/datastructures/any.hpp>
#include <pika/modules/testing.hpp>

#include "small_big_object.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <typeinfo>
#include <utility>

namespace any_tests    // test suite
{
    void test_default_ctor();
    void test_converting_ctor();
    void test_copy_ctor();
    void test_copy_assign();
    void test_converting_assign();
    void test_bad_cast();
    void test_swap_small();
    void test_swap_big();
    void test_null_copying();
    void test_cast_to_reference();

    struct test_case
    {
        char const* const name;
        void (*test_func)();
    };

    const test_case test_cases[] = {{"default construction", test_default_ctor},
        {"single argument construction", test_converting_ctor},
        {"copy construction", test_copy_ctor},
        {"copy assignment operator", test_copy_assign},
        {"converting assignment operator", test_converting_assign},
        {"failed custom keyword cast", test_bad_cast},
        {"swap member function, small", test_swap_small},
        {"swap member function, big", test_swap_big},
        {"copying operations on a null", test_null_copying},
        {"cast to reference types", test_cast_to_reference}};

    using test_case_iterator = test_case const*;

    test_case_iterator begin_tests = test_cases;
    const test_case_iterator end_tests =
        test_cases + (sizeof test_cases / sizeof *test_cases);

    struct copy_counter
    {
    public:
        copy_counter() {}
        copy_counter(const copy_counter&)
        {
            ++count;
        }
        copy_counter& operator=(const copy_counter&)
        {
            ++count;
            return *this;
        }
        static int get_count()
        {
            return count;
        }

    private:
        static int count;
    };

    bool operator==(copy_counter const&, copy_counter const&)
    {
        return true;
    }

    int copy_counter::count = 0;
}    // namespace any_tests

namespace std {

    std::ostream& operator<<(std::ostream& os, std::type_info const&)
    {
        return os;
    }

    std::ostream& operator<<(std::ostream& os, any_tests::copy_counter const&)
    {
        return os;
    }

    std::istream& operator>>(std::istream& is, any_tests::copy_counter&)
    {
        return is;
    }
}    // namespace std

namespace any_tests    // test definitions
{
    void test_default_ctor()
    {
        const pika::any_nonser value;

        PIKA_TEST_MSG(!value.has_value(), "empty");
        PIKA_TEST_EQ_MSG(static_cast<void*>(nullptr), pika::any_cast<int>(&value),
            "pika::any_cast<int>");
        PIKA_TEST_EQ_MSG(
            value.type(), typeid(pika::util::detail::any::empty), "type");
    }

    void test_converting_ctor()
    {
        std::string text = "test message";
        pika::any_nonser value = pika::any_nonser(text);

        PIKA_TEST_EQ_MSG(true, value.has_value(), "empty");
        PIKA_TEST_EQ_MSG(value.type(), typeid(std::string), "type");
        PIKA_TEST_EQ_MSG(static_cast<void*>(nullptr), pika::any_cast<int>(&value),
            "pika::any_cast<int>");
        PIKA_TEST_NEQ_MSG(static_cast<void*>(nullptr),
            pika::any_cast<std::string>(&value), "pika::any_cast<std::string>");
        PIKA_TEST_EQ_MSG(pika::any_cast<std::string>(value), text,
            "comparing cast copy against original text");
        PIKA_TEST_NEQ_MSG(pika::any_cast<std::string>(&value), &text,
            "comparing address in copy against original text");
    }

    void test_copy_ctor()
    {
        std::string text = "test message";
        pika::any_nonser original = pika::any_nonser(text),
                        copy = pika::any_nonser(original);

        PIKA_TEST_EQ_MSG(true, copy.has_value(), "empty");
        PIKA_TEST_EQ_MSG(original.type(), copy.type(), "type");
        PIKA_TEST_EQ_MSG(pika::any_cast<std::string>(original),
            pika::any_cast<std::string>(copy),
            "comparing cast copy against original");
        PIKA_TEST_EQ_MSG(text, pika::any_cast<std::string>(copy),
            "comparing cast copy against original text");
        PIKA_TEST_NEQ_MSG(pika::any_cast<std::string>(&original),
            pika::any_cast<std::string>(&copy),
            "comparing address in copy against original");
    }

    void test_copy_assign()
    {
        std::string text = "test message";
        pika::any_nonser original = pika::any_nonser(text), copy;
        pika::any_nonser* assign_result = &(copy = original);

        PIKA_TEST_EQ_MSG(true, copy.has_value(), "empty");
        PIKA_TEST_EQ_MSG(original.type(), copy.type(), "type");
        PIKA_TEST_EQ_MSG(pika::any_cast<std::string>(original),
            pika::any_cast<std::string>(copy),
            "comparing cast copy against cast original");
        PIKA_TEST_EQ_MSG(text, pika::any_cast<std::string>(copy),
            "comparing cast copy against original text");
        PIKA_TEST_NEQ_MSG(pika::any_cast<std::string>(&original),
            pika::any_cast<std::string>(&copy),
            "comparing address in copy against original");
        PIKA_TEST_EQ_MSG(assign_result, &copy, "address of assignment result");
    }

    void test_converting_assign()
    {
        std::string text = "test message";
        pika::any_nonser value;
        pika::any_nonser* assign_result = &(value = text);

        PIKA_TEST_EQ_MSG(true, value.has_value(), "type");
        PIKA_TEST_EQ_MSG(value.type(), typeid(std::string), "type");
        PIKA_TEST_EQ_MSG(static_cast<void*>(nullptr), pika::any_cast<int>(&value),
            "pika::any_cast<int>");
        PIKA_TEST_NEQ_MSG(static_cast<void*>(nullptr),
            pika::any_cast<std::string>(&value), "pika::any_cast<std::string>");
        PIKA_TEST_EQ_MSG(pika::any_cast<std::string>(value), text,
            "comparing cast copy against original text");
        PIKA_TEST_NEQ_MSG(pika::any_cast<std::string>(&value), &text,
            "comparing address in copy against original text");
        PIKA_TEST_EQ_MSG(assign_result, &value, "address of assignment result");
    }

    void test_bad_cast()
    {
        std::string text = "test message";
        pika::any_nonser value = pika::any_nonser(text);

        {
            bool caught_exception = false;
            try
            {
                pika::any_cast<const char*>(value);
            }
            catch (pika::bad_any_cast const&)
            {
                caught_exception = true;
            }
            catch (...)
            {
                PIKA_TEST_MSG(false, "caught wrong exception");
            }
            PIKA_TEST(caught_exception);
        }
    }

    void test_swap_small()
    {
        if (sizeof(small_object) <= sizeof(void*))
            std::cout << "object is small\n";
        else
            std::cout << "object is large\n";

        small_object text = 17;
        pika::any_nonser original = pika::any_nonser(text), swapped;
        small_object* original_ptr = pika::any_cast<small_object>(&original);
        pika::any_nonser* swap_result = &original.swap(swapped);

        PIKA_TEST_MSG(!original.has_value(), "empty on original");
        PIKA_TEST_EQ_MSG(true, swapped.has_value(), "empty on swapped");
        PIKA_TEST_EQ_MSG(swapped.type(), typeid(small_object), "type");
        PIKA_TEST_EQ_MSG(text, pika::any_cast<small_object>(swapped),
            "comparing swapped copy against original text");
        PIKA_TEST_NEQ_MSG(static_cast<void*>(nullptr), original_ptr,
            "address in pre-swapped original");
        PIKA_TEST_NEQ_MSG(original_ptr, pika::any_cast<small_object>(&swapped),
            "comparing address in swapped against original");
        PIKA_TEST_EQ_MSG(swap_result, &original, "address of swap result");

        pika::any_nonser copy1 = pika::any_nonser(copy_counter());
        pika::any_nonser copy2 = pika::any_nonser(copy_counter());
        int count = copy_counter::get_count();
        swap(copy1, copy2);
        PIKA_TEST_EQ_MSG(count, copy_counter::get_count(),
            "checking that free swap doesn't make pika::any_nonser copies.");
    }

    void test_swap_big()
    {
        if (sizeof(big_object) <= sizeof(void*))
            std::cout << "object is small\n";
        else
            std::cout << "object is large\n";

        big_object text(5, 12);
        pika::any_nonser original = pika::any_nonser(text), swapped;
        big_object* original_ptr = pika::any_cast<big_object>(&original);
        pika::any_nonser* swap_result = &original.swap(swapped);

        PIKA_TEST_MSG(!original.has_value(), "empty on original");
        PIKA_TEST_EQ_MSG(true, swapped.has_value(), "empty on swapped");
        PIKA_TEST_EQ_MSG(swapped.type(), typeid(big_object), "type");
        PIKA_TEST_EQ_MSG(text, pika::any_cast<big_object>(swapped),
            "comparing swapped copy against original text");
        PIKA_TEST_NEQ_MSG(static_cast<void*>(nullptr), original_ptr,
            "address in pre-swapped original");
        PIKA_TEST_EQ_MSG(original_ptr, pika::any_cast<big_object>(&swapped),
            "comparing address in swapped against original");
        PIKA_TEST_EQ_MSG(swap_result, &original, "address of swap result");

        pika::any_nonser copy1 = pika::any_nonser(copy_counter());
        pika::any_nonser copy2 = pika::any_nonser(copy_counter());
        int count = copy_counter::get_count();
        swap(copy1, copy2);
        PIKA_TEST_EQ_MSG(count, copy_counter::get_count(),
            "checking that free swap doesn't make pika::any_nonser copies.");
    }

    void test_null_copying()
    {
        const pika::any_nonser null;
        pika::any_nonser copied = null, assigned;
        assigned = null;

        PIKA_TEST_MSG(!null.has_value(), "empty on null");
        PIKA_TEST_MSG(!copied.has_value(), "empty on copied");
        PIKA_TEST_MSG(!assigned.has_value(), "empty on copied");
    }

    void test_cast_to_reference()
    {
        pika::any_nonser a(137);
        const pika::any_nonser b(a);

        int& ra = pika::any_cast<int&>(a);
        int const& ra_c = pika::any_cast<int const&>(a);
        int volatile& ra_v = pika::any_cast<int volatile&>(a);
        int const volatile& ra_cv = pika::any_cast<int const volatile&>(a);

        PIKA_TEST_MSG(&ra == &ra_c && &ra == &ra_v && &ra == &ra_cv,
            "cv references to same obj");

        int const& rb_c = pika::any_cast<int const&>(b);
        int const volatile& rb_cv = pika::any_cast<int const volatile&>(b);

        PIKA_TEST_MSG(&rb_c == &rb_cv, "cv references to copied const obj");
        PIKA_TEST_MSG(&ra != &rb_c, "copies hold different objects");

        ++ra;
        int incremented = pika::any_cast<int>(a);
        PIKA_TEST_MSG(
            incremented == 138, "increment by reference changes value");

        {
            bool caught_exception = false;
            try
            {
                pika::any_cast<char&>(a);
            }
            catch (pika::bad_any_cast const&)
            {
                caught_exception = true;
            }
            catch (...)
            {
                PIKA_TEST_MSG(false, "caught wrong exception");
            }
            PIKA_TEST(caught_exception);
        }

        {
            bool caught_exception = false;
            try
            {
                pika::any_cast<const char&>(b);
            }
            catch (pika::bad_any_cast const&)
            {
                caught_exception = true;
            }
            catch (...)
            {
                PIKA_TEST_MSG(false, "caught wrong exception");
            }
            PIKA_TEST(caught_exception);
        }
    }
}    // namespace any_tests

int main()
{
    using namespace any_tests;
    while (begin_tests != end_tests)
    {
        (*begin_tests->test_func)();
        ++begin_tests;
    }
    return pika::util::report_errors();
}
