//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
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
            const char* name() const noexcept { return "pika"; }

            std::string message(int value) const
            {
                if (value >= static_cast<int>(pika::error::success) &&
                    value < static_cast<int>(pika::error::last_error))
                    return std::string("pika(") + error_names[value] + ")";    //-V108
                if (error_code_has_system_error(value)) return std::string("pika(system_error)");
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
            const char* name() const noexcept { return ""; }

            std::string message(int) const noexcept { return ""; }
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

    namespace detail {
        std::error_category const& get_lightweight_pika_category()
        {
            static detail::lightweight_pika_category lightweight_pika_category;
            return lightweight_pika_category;
        }

        std::error_category const& get_pika_category(throwmode mode)
        {
            switch (mode)
            {
            case throwmode::rethrow: return get_pika_rethrow_category();

            case throwmode::lightweight:
            case throwmode::lightweight_rethrow: return get_lightweight_pika_category();

            case throwmode::plain:
            default: break;
            }
            return pika::get_pika_category();
        }

        bool throwmode_is_lightweight(throwmode mode)
        {
            return static_cast<int>(mode) & static_cast<int>(throwmode::lightweight);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    error_code::error_code(error e, throwmode mode)
      : std::error_code(detail::make_system_error_code(e, mode))
    {
        if (e != pika::error::success && e != pika::error::no_success &&
            !(detail::throwmode_is_lightweight(mode)))
            exception_ = detail::get_exception(e, "", mode);
    }

    error_code::error_code(error e, char const* func, char const* file, long line, throwmode mode)
      : std::error_code(detail::make_system_error_code(e, mode))
    {
        if (e != pika::error::success && e != pika::error::no_success &&
            !(detail::throwmode_is_lightweight(mode)))
        {
            exception_ = detail::get_exception(e, "", mode, func, file, line);
        }
    }

    error_code::error_code(error e, char const* msg, throwmode mode)
      : std::error_code(detail::make_system_error_code(e, mode))
    {
        if (e != pika::error::success && e != pika::error::no_success &&
            !(detail::throwmode_is_lightweight(mode)))
            exception_ = detail::get_exception(e, msg, mode);
    }

    error_code::error_code(
        error e, char const* msg, char const* func, char const* file, long line, throwmode mode)
      : std::error_code(detail::make_system_error_code(e, mode))
    {
        if (e != pika::error::success && e != pika::error::no_success &&
            !(detail::throwmode_is_lightweight(mode)))
        {
            exception_ = detail::get_exception(e, msg, mode, func, file, line);
        }
    }

    error_code::error_code(error e, std::string const& msg, throwmode mode)
      : std::error_code(detail::make_system_error_code(e, mode))
    {
        if (e != pika::error::success && e != pika::error::no_success &&
            !(detail::throwmode_is_lightweight(mode)))
            exception_ = detail::get_exception(e, msg, mode);
    }

    error_code::error_code(error e, std::string const& msg, char const* func, char const* file,
        long line, throwmode mode)
      : std::error_code(detail::make_system_error_code(e, mode))
    {
        if (e != pika::error::success && e != pika::error::no_success &&
            !(detail::throwmode_is_lightweight(mode)))
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
      : std::error_code(detail::make_system_error_code(get_error(e), throwmode::rethrow))
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
      : std::error_code(static_cast<pika::error>(rhs.value()) == pika::error::success ?
                make_success_code((category() == detail::get_lightweight_pika_category()) ?
                        pika::throwmode::lightweight :
                        pika::throwmode::plain) :
                rhs)
      , exception_(rhs.exception_)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    error_code& error_code::operator=(error_code const& rhs)
    {
        if (this != &rhs)
        {
            if (static_cast<pika::error>(rhs.value()) == pika::error::success)
            {
                // if the rhs is a success code, we maintain our throw mode
                this->std::error_code::operator=(
                    make_success_code((category() == detail::get_lightweight_pika_category()) ?
                            pika::throwmode::lightweight :
                            pika::throwmode::plain));
            }
            else { this->std::error_code::operator=(rhs); }
            exception_ = rhs.exception_;
        }
        return *this;
    }
    /// \endcond
}    // namespace pika
