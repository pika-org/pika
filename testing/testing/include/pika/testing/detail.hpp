////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/functional/function.hpp>
#include <pika/preprocessor/cat.hpp>
#include <pika/preprocessor/expand.hpp>
#include <pika/preprocessor/nargs.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/thread_support/spinlock.hpp>
#include <pika/util/ios_flags_saver.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/std.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <ostream>
#endif

namespace pika::detail {
    enum counter_type
    {
        counter_sanity,
        counter_test
    };

    // Helper to wrap pointers in fmt::ptr so that users of PIKA_TEST* don't
    // have to wrap values to be tested in fmt::ptr.
    template <typename T>
    auto make_fmt_ptr(T* x)
    {
        return fmt::ptr(x);
    }

    template <typename T>
    T& make_fmt_ptr(T& x)
    {
        return x;
    }

    struct fixture final
    {
    public:
        using mutex_type = pika::detail::spinlock;

    private:
        std::ostream& stream_;
        static std::atomic<std::size_t> sanity_tests_;
        static std::atomic<std::size_t> sanity_failures_;
        static std::atomic<std::size_t> test_tests_;
        static std::atomic<std::size_t> test_failures_;
        mutex_type mutex_;

    public:
        explicit fixture(std::ostream& stream);
        ~fixture();

        PIKA_EXPORT void increment_tests(counter_type c);
        PIKA_EXPORT void increment_failures(counter_type c);

        PIKA_EXPORT std::size_t get_tests(counter_type c) const;
        PIKA_EXPORT std::size_t get_failures(counter_type c) const;

        template <typename T>
        bool check_(char const* file, int line, char const* function, counter_type c, T const& t,
            char const* msg)
        {
            increment_tests(c);

            if (!t)
            {
                std::lock_guard<mutex_type> l(mutex_);
                // nvc++ is not able to compile this with a stream as the
                // first argument. As a workaround we leave the stream_
                // argument out and print to the default stream. The same
                // workaround is applied elsewhere in this file.
                fmt::print(
#if !defined(PIKA_NVHPC_VERSION)
                    stream_,
#endif
                    "{}({}): {} failed in function '{}'\n", file, line, msg, function);
                increment_failures(c);
                return false;
            }
            return true;
        }

        template <typename T, typename U>
        bool check_equal(char const* file, int line, char const* function, counter_type c,
            T const& t, U const& u, char const* msg)
        {
            increment_tests(c);

            if (!(t == u))
            {
                std::lock_guard<mutex_type> l(mutex_);
                fmt::print(
#if !defined(PIKA_NVHPC_VERSION)
                    stream_,
#endif
                    "{}({}): {} failed in function '{}': '{} != {}'\n", file, line, msg, function,
                    make_fmt_ptr(t), make_fmt_ptr(u));
                increment_failures(c);
                return false;
            }
            return true;
        }

        template <typename T, typename U>
        bool check_not_equal(char const* file, int line, char const* function, counter_type c,
            T const& t, U const& u, char const* msg)
        {
            increment_tests(c);

            if (!(t != u))
            {
                std::lock_guard<mutex_type> l(mutex_);
                fmt::print(
#if !defined(PIKA_NVHPC_VERSION)
                    stream_,
#endif
                    "{}({}): {} failed in function '{}': '{} == {}'\n", file, line, msg, function,
                    make_fmt_ptr(t), make_fmt_ptr(u));
                increment_failures(c);
                return false;
            }
            return true;
        }

        template <typename T, typename U>
        bool check_less(char const* file, int line, char const* function, counter_type c,
            T const& t, U const& u, char const* msg)
        {
            increment_tests(c);

            if (!(t < u))
            {
                std::lock_guard<mutex_type> l(mutex_);
                fmt::print(
#if !defined(PIKA_NVHPC_VERSION)
                    stream_,
#endif
                    "{}({}): {} failed in function '{}': '{} >= {}'\n", file, line, msg, function,
                    make_fmt_ptr(t), make_fmt_ptr(u));
                increment_failures(c);
                return false;
            }
            return true;
        }

        template <typename T, typename U>
        bool check_less_equal(char const* file, int line, char const* function, counter_type c,
            T const& t, U const& u, char const* msg)
        {
            increment_tests(c);

            if (!(t <= u))
            {
                std::lock_guard<mutex_type> l(mutex_);
                pika::detail::ios_flags_saver ifs(stream_);
                fmt::print(
#if !defined(PIKA_NVHPC_VERSION)
                    stream_,
#endif
                    "{}({}): {} failed in function '{}': '{} > {}'\n", file, line, msg, function,
                    make_fmt_ptr(t), make_fmt_ptr(u));
                increment_failures(c);
                return false;
            }
            return true;
        }

        template <typename T, typename U, typename V>
        bool check_range(char const* file, int line, char const* function, counter_type c,
            T const& t, U const& u, V const& v, char const* msg)
        {
            increment_tests(c);

            if (!(t >= u && t <= v))
            {
                std::lock_guard<mutex_type> l(mutex_);
                if (!(t >= u))
                {
                    std::lock_guard<mutex_type> l(mutex_);
                    fmt::print(
#if !defined(PIKA_NVHPC_VERSION)
                        stream_,
#endif
                        "{}({}): {} failed in function '{}': '{} > {}'\n", file, line, msg,
                        function, make_fmt_ptr(t), make_fmt_ptr(u));
                    increment_failures(c);
                    return false;
                }
                else
                {
                    fmt::print(
#if !defined(PIKA_NVHPC_VERSION)
                        stream_,
#endif
                        "{}({}): {} failed in function '{}': '{} > {}'\n", file, line, msg,
                        function, make_fmt_ptr(t), u);
                }
                increment_failures(c);
                return false;
            }
            return true;
        }
    };

    PIKA_EXPORT extern fixture global_fixture;

    ////////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT int report_errors();
    PIKA_EXPORT int report_errors(std::ostream& stream);
}    // namespace pika::detail
