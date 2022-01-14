//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>

#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)

#include <pika/serialization/input_archive.hpp>
#include <pika/serialization/output_archive.hpp>
#include <pika/serialization/serialize.hpp>
#include <pika/serialization/string.hpp>
#include <pika/serialization/variant.hpp>

#include <pika/modules/testing.hpp>

#include <string>
#include <vector>

#include <boost/variant.hpp>

template <typename T>
struct A
{
    A() {}

    explicit A(T t)
      : t_(t)
    {
    }
    T t_;

    A& operator=(T t)
    {
        t_ = t;
        return *this;
    }

    friend bool operator==(A a, A b)
    {
        return a.t_ == b.t_;
    }

    friend std::ostream& operator<<(std::ostream& os, A a)
    {
        os << a.t_;
        return os;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar& t_;
    }
};

#endif

int main()
{
#if defined(PIKA_SERIALIZATION_HAVE_BOOST_TYPES)
    std::vector<char> buf;
    pika::serialization::output_archive oar(buf);
    pika::serialization::input_archive iar(buf);

    boost::variant<int, std::string, double, A<int>> ovar =
        std::string("dfsdf");
    boost::variant<int, std::string, double, A<int>> ivar;
    oar << ovar;
    iar >> ivar;

    PIKA_TEST_EQ(ivar.which(), ovar.which());
    PIKA_TEST_EQ(ivar, ovar);

    ovar = 2.5;
    oar << ovar;
    iar >> ivar;

    PIKA_TEST_EQ(ivar.which(), ovar.which());
    PIKA_TEST_EQ(ivar, ovar);

    ovar = 1;
    oar << ovar;
    iar >> ivar;

    PIKA_TEST_EQ(ivar.which(), ovar.which());
    PIKA_TEST_EQ(ivar, ovar);

    ovar = A<int>(2);
    oar << ovar;
    iar >> ivar;

    PIKA_TEST_EQ(ivar.which(), ovar.which());
    PIKA_TEST_EQ(ivar, ovar);

    const boost::variant<std::string> sovar = std::string("string");
    boost::variant<std::string> sivar;
    oar << sovar;
    iar >> sivar;

    PIKA_TEST_EQ(sivar.which(), sovar.which());
    PIKA_TEST_EQ(sivar, sovar);

    return pika::util::report_errors();
#else
    return 0;
#endif
}
