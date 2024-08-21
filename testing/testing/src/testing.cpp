//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(PIKA_HAVE_MODULE)
module;
#endif

#define PIKA_NO_VERSION_CHECK

#include <pika/assert.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/modules/util.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.testing;
#endif

namespace pika::detail {

    std::atomic<std::size_t> fixture::sanity_tests_(0);
    std::atomic<std::size_t> fixture::sanity_failures_(0);
    std::atomic<std::size_t> fixture::test_tests_(0);
    std::atomic<std::size_t> fixture::test_failures_(0);

    fixture::fixture(std::ostream& stream)
      : stream_(stream)
    {
    }

    fixture::~fixture()
    {
        if (report_errors(stream_) != 0) {
            // std::exit(EXIT_FAILURE);
            std::exit(1);
        }
    }

    void fixture::increment_tests(counter_type c)
    {
        switch (c)
        {
        case counter_sanity: ++sanity_tests_; return;
        case counter_test: ++test_tests_; return;
        default: break;
        }
        PIKA_ASSERT(false);
    }

    void fixture::increment_failures(counter_type c)
    {
        switch (c)
        {
        case counter_sanity: ++sanity_failures_; return;
        case counter_test: ++test_failures_; return;
        default: break;
        }
        PIKA_ASSERT(false);
    }

    std::size_t fixture::get_tests(counter_type c) const
    {
        switch (c)
        {
        case counter_sanity: return sanity_tests_;
        case counter_test: return test_tests_;
        default: break;
        }
        PIKA_ASSERT(false);
        return std::size_t(-1);
    }

    std::size_t fixture::get_failures(counter_type c) const
    {
        switch (c)
        {
        case counter_sanity: return sanity_failures_;
        case counter_test: return test_failures_;
        default: break;
        }
        PIKA_ASSERT(false);
        return std::size_t(-1);
    }

    fixture global_fixture{std::cerr};

    int report_errors() { return report_errors(std::cerr); }

    int report_errors(std::ostream& stream)
    {
        auto sanity_tests = detail::global_fixture.get_tests(counter_sanity);
        auto test_tests = detail::global_fixture.get_tests(counter_test);
        auto sanity_failures = detail::global_fixture.get_failures(counter_sanity);
        auto test_failures = detail::global_fixture.get_failures(counter_test);

        if (sanity_tests == 0 && test_tests == 0)
        {
            pika::detail::ios_flags_saver ifs(stream);
            stream << "No tests run. Did you forget to add tests?" << std::endl;
            return 1;
        }
        else if (sanity_failures == 0 && test_failures == 0)
        {
            pika::detail::ios_flags_saver ifs(stream);
            stream << "All tests passed. Ran " << sanity_tests << " sanity check"    //-V128
                   << ((sanity_tests == 1) ? " and " : "s and ") << test_tests << " test"
                   << ((test_tests == 1) ? "." : "s.") << std::endl;
            return 0;
        }
        else
        {
            pika::detail::ios_flags_saver ifs(stream);
            stream << "Tests failed. " << sanity_failures << "/" << sanity_tests
                   << " sanity check"    //-V128
                   << ((sanity_tests == 1) ? " and " : "s and ") << test_failures << "/"
                   << test_tests << " test" << ((test_tests == 1) ? " failed." : "s failed.")
                   << std::endl;
            return 1;
        }
    }
}    // namespace pika::detail
