//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/errors/exception.hpp>
#include <pika/errors/exception_list.hpp>
#include <pika/thread_support/unlock_guard.hpp>

#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace pika {
    namespace detail {
        std::string indent_message(std::string const& msg_)
        {
            std::string result;
            std::string const& msg(msg_);
            std::string::size_type pos = msg.find_first_of('\n');
            std::string::size_type first_non_ws = msg.find_first_not_of(" \n");
            std::string::size_type pos1 = 0;

            while (std::string::npos != pos)
            {
                if (pos > first_non_ws)
                {    // skip leading newline
                    result += msg.substr(pos1, pos - pos1 + 1);
                    pos = msg.find_first_of('\n', pos1 = pos + 1);
                    if (std::string::npos != pos)
                    {
                        result += "  ";
                    }
                }
                else
                {
                    pos = msg.find_first_of('\n', pos1 = pos + 1);
                }
            }

            result += msg.substr(pos1);
            return result;
        }
    }    // namespace detail

    error_code throws;    // "throw on error" special error_code;
                          //
                          // Note that it doesn't matter if this isn't
                          // initialized before use since the only use is
                          // to take its address for comparison purposes.

    exception_list::exception_list()
      : pika::exception(pika::success)
      , mtx_()
    {
    }

    exception_list::exception_list(std::exception_ptr const& e)
      : pika::exception(pika::get_error(e), pika::get_error_what(e))
      , mtx_()
    {
        add_no_lock(e);
    }

    exception_list::exception_list(exception_list_type&& l)
      : pika::exception(!l.empty() ? pika::get_error(l.front()) : success)
      , exceptions_(PIKA_MOVE(l))
      , mtx_()
    {
    }

    exception_list::exception_list(exception_list const& l)
      : pika::exception(static_cast<pika::exception const&>(l))
      , exceptions_(l.exceptions_)
      , mtx_()
    {
    }

    exception_list::exception_list(exception_list&& l)
      : pika::exception(PIKA_MOVE(static_cast<pika::exception&>(l)))
      , exceptions_(PIKA_MOVE(l.exceptions_))
      , mtx_()
    {
    }

    exception_list& exception_list::operator=(exception_list const& l)
    {
        if (this != &l)
        {
            *static_cast<pika::exception*>(this) =
                static_cast<pika::exception const&>(l);
            exceptions_ = l.exceptions_;
        }
        return *this;
    }

    exception_list& exception_list::operator=(exception_list&& l)
    {
        if (this != &l)
        {
            static_cast<pika::exception&>(*this) =
                PIKA_MOVE(static_cast<pika::exception&>(l));
            exceptions_ = PIKA_MOVE(l.exceptions_);
        }
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::error_code exception_list::get_error() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (exceptions_.empty())
            return pika::no_success;
        return pika::get_error(exceptions_.front());
    }

    std::string exception_list::get_message() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (exceptions_.empty())
            return "";

        if (1 == exceptions_.size())
            return pika::get_error_what(exceptions_.front());

        std::string result("\n");

        exception_list_type::const_iterator end = exceptions_.end();
        exception_list_type::const_iterator it = exceptions_.begin();
        for (/**/; it != end; ++it)
        {
            result += "  ";
            result += detail::indent_message(pika::get_error_what(*it));
            if (result.find_last_of('\n') < result.size() - 1)
                result += "\n";
        }
        return result;
    }

    void exception_list::add(std::exception_ptr const& e)
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (exceptions_.empty())
        {
            pika::exception ex;
            {
                util::unlock_guard<std::unique_lock<mutex_type>> ul(l);
                ex = pika::exception(pika::get_error(e));
            }

            // set the error code for our base class
            static_cast<pika::exception&>(*this) = ex;
        }
        exceptions_.push_back(e);
    }

    void exception_list::add_no_lock(std::exception_ptr const& e)
    {
        if (exceptions_.empty())
        {
            // set the error code for our base class
            static_cast<pika::exception&>(*this) =
                pika::exception(pika::get_error(e));
        }
        exceptions_.push_back(e);
    }
}    // namespace pika
