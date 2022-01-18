////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <pika/datastructures/any.hpp>
#include <pika/modules/testing.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "small_big_object.hpp"

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        pika::util::streamable_any_nonser any1(big_object(30, 40));
        std::stringstream buffer;

        buffer << any1;

        PIKA_TEST(buffer.str() == "3040");
    }

    // non serializable version
    {
        // test equality
        {
            pika::any_nonser any1_nonser(7), any2_nonser(7), any3_nonser(10),
                any4_nonser(std::string("seven"));

            PIKA_TEST_EQ(pika::any_cast<int>(any1_nonser), 7);
            PIKA_TEST_NEQ(pika::any_cast<int>(any1_nonser), 10);
            PIKA_TEST_NEQ(pika::any_cast<int>(any1_nonser), 10.0f);
            PIKA_TEST_EQ(pika::any_cast<int>(any1_nonser),
                pika::any_cast<int>(any1_nonser));
            PIKA_TEST_EQ(pika::any_cast<int>(any1_nonser),
                pika::any_cast<int>(any2_nonser));
            PIKA_TEST(any1_nonser.type() == any3_nonser.type());
            PIKA_TEST(any1_nonser.type() != any4_nonser.type());

            std::string long_str =
                std::string("This is a looooooooooooooooooooooooooong string");
            std::string other_str = std::string("a different string");
            any1_nonser = long_str;
            any2_nonser = any1_nonser;
            any3_nonser = other_str;
            any4_nonser = 10.0f;

            PIKA_TEST_EQ(pika::any_cast<std::string>(any1_nonser), long_str);
            PIKA_TEST_NEQ(pika::any_cast<std::string>(any1_nonser), other_str);
            PIKA_TEST(any1_nonser.type() == typeid(std::string));
            PIKA_TEST(pika::any_cast<std::string>(any1_nonser) ==
                pika::any_cast<std::string>(any1_nonser));
            PIKA_TEST(pika::any_cast<std::string>(any1_nonser) ==
                pika::any_cast<std::string>(any2_nonser));
            PIKA_TEST(any1_nonser.type() == any3_nonser.type());
            PIKA_TEST(any1_nonser.type() != any4_nonser.type());
        }

        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            pika::any_nonser any1_nonser(f);
            pika::any_nonser any2_nonser(any1_nonser);
            pika::any_nonser any3_nonser = any1_nonser;

            PIKA_TEST_EQ((pika::any_cast<small_object>(any1_nonser))(2),
                uint64_t(17 + 2));
            PIKA_TEST_EQ((pika::any_cast<small_object>(any2_nonser))(4),
                uint64_t(17 + 4));
            PIKA_TEST_EQ((pika::any_cast<small_object>(any3_nonser))(6),
                uint64_t(17 + 6));
        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            pika::any_nonser any1_nonser(f);
            pika::any_nonser any2_nonser(any1_nonser);
            pika::any_nonser any3_nonser = any1_nonser;

            PIKA_TEST_EQ((pika::any_cast<big_object>(any1_nonser))(3, 4),
                uint64_t(5 + 12 + 3 + 4));
            PIKA_TEST_EQ((pika::any_cast<big_object>(any2_nonser))(5, 6),
                uint64_t(5 + 12 + 5 + 6));
            PIKA_TEST_EQ((pika::any_cast<big_object>(any3_nonser))(7, 8),
                uint64_t(5 + 12 + 7 + 8));
        }

        // move semantics
        {
            pika::any_nonser any1(5);
            PIKA_TEST(any1.has_value());
            pika::any_nonser any2(std::move(any1));
            PIKA_TEST(any2.has_value());
            PIKA_TEST(!any1.has_value());    // NOLINT
        }

        {
            pika::any_nonser any1(5);
            PIKA_TEST(any1.has_value());
            pika::any_nonser any2;
            PIKA_TEST(!any2.has_value());

            any2 = std::move(any1);
            PIKA_TEST(any2.has_value());
            PIKA_TEST(!any1.has_value());    // NOLINT
        }
    }

    return pika::util::report_errors();
}
