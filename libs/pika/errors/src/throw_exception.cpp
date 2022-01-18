//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/errors/error.hpp>
#include <pika/errors/exception.hpp>
#include <pika/modules/filesystem.hpp>

#include <exception>
#include <string>
#include <system_error>

namespace pika { namespace detail {
    PIKA_NORETURN void throw_exception(error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line)
    {
        filesystem::path p(file);
        pika::detail::throw_exception(
            pika::exception(errcode, msg, pika::plain), func, p.string(), line);
    }

    PIKA_NORETURN void rethrow_exception(
        exception const& e, std::string const& func)
    {
        pika::detail::throw_exception(
            pika::exception(e.get_error(), e.what(), pika::rethrow), func,
            pika::get_error_file_name(e), pika::get_error_line_number(e));
    }

    std::exception_ptr get_exception(error errcode, std::string const& msg,
        throwmode mode, std::string const& /* func */, std::string const& file,
        long line, std::string const& auxinfo)
    {
        filesystem::path p(file);
        return pika::detail::get_exception(pika::exception(errcode, msg, mode),
            p.string(), file, line, auxinfo);
    }

    std::exception_ptr get_exception(std::error_code const& ec,
        std::string const& /* msg */, throwmode /* mode */,
        std::string const& func, std::string const& file, long line,
        std::string const& auxinfo)
    {
        return pika::detail::get_exception(
            pika::exception(ec), func, file, line, auxinfo);
    }

    void throws_if(pika::error_code& ec, error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line)
    {
        if (&ec == &pika::throws)
        {
            pika::detail::throw_exception(errcode, msg, func, file, line);
        }
        else
        {
            ec = make_error_code(static_cast<pika::error>(errcode), msg,
                func.c_str(), file.c_str(), line,
                (ec.category() == pika::get_lightweight_pika_category()) ?
                    pika::lightweight :
                    pika::plain);
        }
    }

    void rethrows_if(
        pika::error_code& ec, exception const& e, std::string const& func)
    {
        if (&ec == &pika::throws)
        {
            pika::detail::rethrow_exception(e, func);
        }
        else
        {
            ec = make_error_code(e.get_error(), e.what(), func.c_str(),
                pika::get_error_file_name(e).c_str(),
                pika::get_error_line_number(e),
                (ec.category() == pika::get_lightweight_pika_category()) ?
                    pika::lightweight_rethrow :
                    pika::rethrow);
        }
    }

    PIKA_NORETURN void throw_thread_interrupted_exception()
    {
        throw pika::thread_interrupted();
    }
}}    // namespace pika::detail
