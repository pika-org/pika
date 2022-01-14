//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Checking that #582 was fixed

#include <pika/local/exception.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>

int pika_main()
{
    PIKA_THROW_EXCEPTION(pika::invalid_status, "pika_main", "testing");
    return pika::local::finalize();
}

int main(int argc, char** argv)
{
    bool caught_exception = false;
    try
    {
        pika::local::init(pika_main, argc, argv);
    }
    catch (pika::exception const& e)
    {
        PIKA_TEST(e.get_error() == pika::invalid_status);
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }
    PIKA_TEST(caught_exception);

    return pika::util::report_errors();
}
