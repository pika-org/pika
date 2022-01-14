//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/errors/error_code.hpp>
#include <pika/errors/exception.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    namespace detail {
        class pika_category : public std::error_category
        {
        public:
            const char* name() const noexcept
            {
                return "pika";
            }

            std::string message(int value) const
            {
                if (value >= success && value < last_error)
                    return std::string("pika(") + error_names[value] +
                        ")";    //-V108
                if (value & system_error_flag)
                    return std::string("pika(system_error)");
                return "pika(unknown_error)";
            }
        };

        struct lightweight_pika_category : pika_category
        {
        };

        // this doesn't add any text to the exception what() message
        class pika_category_rethrow : public std::error_category
        {
        public:
            const char* name() const noexcept
            {
                return "";
            }

            std::string message(int) const noexcept
            {
                return "";
            }
        };

        struct lightweight_pika_category_rethrow : pika_category_rethrow
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    std::error_category const& get_pika_category()
    {
        static detail::pika_category pika_category;
        return pika_category;
    }

    std::error_category const& get_pika_rethrow_category()
    {
        static detail::pika_category_rethrow pika_category_rethrow;
        return pika_category_rethrow;
    }

    std::error_category const& get_lightweight_pika_category()
    {
        static detail::lightweight_pika_category lightweight_pika_category;
        return lightweight_pika_category;
    }

    std::error_category const& get_pika_category(throwmode mode)
    {
        switch (mode)
        {
        case rethrow:
            return get_pika_rethrow_category();

        case lightweight:
        case lightweight_rethrow:
            return get_lightweight_pika_category();

        case plain:
        default:
            break;
        }
        return get_pika_category();
    }

    ///////////////////////////////////////////////////////////////////////////
    error_code::error_code(error e, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(e, "", mode);
    }

    error_code::error_code(
        error e, char const* func, char const* file, long line, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
        {
            exception_ = detail::get_exception(e, "", mode, func, file, line);
        }
    }

    error_code::error_code(error e, char const* msg, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(e, msg, mode);
    }

    error_code::error_code(error e, char const* msg, char const* func,
        char const* file, long line, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
        {
            exception_ = detail::get_exception(e, msg, mode, func, file, line);
        }
    }

    error_code::error_code(error e, std::string const& msg, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
            exception_ = detail::get_exception(e, msg, mode);
    }

    error_code::error_code(error e, std::string const& msg, char const* func,
        char const* file, long line, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != success && e != no_success && !(mode & lightweight))
        {
            exception_ = detail::get_exception(e, msg, mode, func, file, line);
        }
    }

    error_code::error_code(int err, pika::exception const& e)
    {
        this->std::error_code::assign(err, get_pika_category());
        exception_ = std::make_exception_ptr(e);
    }

    error_code::error_code(std::exception_ptr const& e)
      : std::error_code(make_system_error_code(get_error(e), rethrow))
      , exception_(e)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string error_code::get_message() const
    {
        if (exception_)
        {
            try
            {
                std::rethrow_exception(exception_);
            }
            catch (std::exception const& be)
            {
                return be.what();
            }
        }
        return get_error_what(*this);    // provide at least minimal error text
    }

    ///////////////////////////////////////////////////////////////////////////
    error_code::error_code(error_code const& rhs)
      : std::error_code(rhs.value() == success ?
                make_success_code(
                    (category() == get_lightweight_pika_category()) ?
                        pika::lightweight :
                        pika::plain) :
                rhs)
      , exception_(rhs.exception_)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    error_code& error_code::operator=(error_code const& rhs)
    {
        if (this != &rhs)
        {
            if (rhs.value() == success)
            {
                // if the rhs is a success code, we maintain our throw mode
                this->std::error_code::operator=(make_success_code(
                    (category() == get_lightweight_pika_category()) ?
                        pika::lightweight :
                        pika::plain));
            }
            else
            {
                this->std::error_code::operator=(rhs);
            }
            exception_ = rhs.exception_;
        }
        return *this;
    }
    /// \endcond
}    // namespace pika
