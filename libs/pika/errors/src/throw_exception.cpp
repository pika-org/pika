//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(PIKA_HAVE_MODULE)
module;
#endif

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/errors/error.hpp>
#include <pika/errors/exception.hpp>

#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.errors;
#endif

namespace pika::detail {
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    [[noreturn]] void throw_exception(error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        std::filesystem::path p(file);
        pika::detail::throw_exception(
            pika::exception(errcode, msg, pika::throwmode::plain), func, p.string(), line);
    }

    [[noreturn]] void rethrow_exception(exception const& e, std::string const& func)
    {
        pika::detail::throw_exception(
            pika::exception(e.get_error(), e.what(), pika::throwmode::rethrow), func,
            pika::get_error_file_name(e), pika::get_error_line_number(e));
    }

    std::exception_ptr get_exception(error errcode, std::string const& msg, throwmode mode,
        std::string const& /* func */, std::string const& file, long line,
        std::string const& auxinfo)
    {
        std::filesystem::path p(file);
        return pika::detail::get_exception(
            pika::exception(errcode, msg, mode), p.string(), file, line, auxinfo);
    }

    std::exception_ptr get_exception(std::error_code const& ec, std::string const& /* msg */,
        throwmode /* mode */, std::string const& func, std::string const& file, long line,
        std::string const& auxinfo)
    {
        return pika::detail::get_exception(pika::exception(ec), func, file, line, auxinfo);
    }

    void throws_if(pika::error_code& ec, error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line)
    {
        if (&ec == &pika::throws) { pika::detail::throw_exception(errcode, msg, func, file, line); }
        else
        {
            ec = make_error_code(static_cast<pika::error>(errcode), msg, func.c_str(), file.c_str(),
                line,
                (ec.category() == get_lightweight_pika_category()) ? pika::throwmode::lightweight :
                                                                     pika::throwmode::plain);
        }
    }

    void rethrows_if(pika::error_code& ec, exception const& e, std::string const& func)
    {
        if (&ec == &pika::throws) { pika::detail::rethrow_exception(e, func); }
        else
        {
            ec = make_error_code(e.get_error(), e.what(), func.c_str(),
                pika::get_error_file_name(e).c_str(), pika::get_error_line_number(e),
                (ec.category() == get_lightweight_pika_category()) ?
                    pika::throwmode::lightweight_rethrow :
                    pika::throwmode::rethrow);
        }
    }

    [[noreturn]] void throw_thread_interrupted_exception() { throw pika::thread_interrupted(); }
}    // namespace pika::detail
