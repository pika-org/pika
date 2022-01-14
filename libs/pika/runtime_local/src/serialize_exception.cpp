//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/modules/errors.hpp>
#include <pika/runtime_local/custom_exception_info.hpp>
#include <pika/runtime_local/detail/serialize_exception.hpp>
#include <pika/serialization/serialize.hpp>

#if ASIO_HAS_BOOST_THROW_EXCEPTION != 0
#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception/exception.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <typeinfo>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace runtime_local { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // TODO: This is not scalable, and painful to update.
    void save_custom_exception(pika::serialization::output_archive& ar,
        std::exception_ptr const& ep, unsigned int /* version */)
    {
        pika::util::exception_type type(pika::util::unknown_exception);
        std::string what;
        int err_value = pika::success;
        std::string err_message;

        std::uint32_t throw_locality_ = 0;
        std::string throw_hostname_;
        std::int64_t throw_pid_ = -1;
        std::size_t throw_shepherd_ = 0;
        std::size_t throw_thread_id_ = 0;
        std::string throw_thread_name_;
        std::string throw_function_;
        std::string throw_file_;
        std::string throw_back_trace_;
        long throw_line_ = 0;
        std::string throw_env_;
        std::string throw_config_;
        std::string throw_state_;
        std::string throw_auxinfo_;

        // retrieve information related to exception_info
        try
        {
            std::rethrow_exception(ep);
        }
        catch (exception_info const& xi)
        {
            std::string const* function = xi.get<pika::detail::throw_function>();
            if (function)
                throw_function_ = *function;

            std::string const* file = xi.get<pika::detail::throw_file>();
            if (file)
                throw_file_ = *file;

            long const* line = xi.get<pika::detail::throw_line>();
            if (line)
                throw_line_ = *line;

            std::uint32_t const* locality =
                xi.get<pika::detail::throw_locality>();
            if (locality)
                throw_locality_ = *locality;

            std::string const* hostname_ =
                xi.get<pika::detail::throw_hostname>();
            if (hostname_)
                throw_hostname_ = *hostname_;

            std::int64_t const* pid_ = xi.get<pika::detail::throw_pid>();
            if (pid_)
                throw_pid_ = *pid_;

            std::size_t const* shepherd = xi.get<pika::detail::throw_shepherd>();
            if (shepherd)
                throw_shepherd_ = *shepherd;

            std::size_t const* thread_id =
                xi.get<pika::detail::throw_thread_id>();
            if (thread_id)
                throw_thread_id_ = *thread_id;

            std::string const* thread_name =
                xi.get<pika::detail::throw_thread_name>();
            if (thread_name)
                throw_thread_name_ = *thread_name;

            std::string const* back_trace =
                xi.get<pika::detail::throw_stacktrace>();
            if (back_trace)
                throw_back_trace_ = *back_trace;

            std::string const* env_ = xi.get<pika::detail::throw_env>();
            if (env_)
                throw_env_ = *env_;

            std::string const* config_ = xi.get<pika::detail::throw_config>();
            if (config_)
                throw_config_ = *config_;

            std::string const* state_ = xi.get<pika::detail::throw_state>();
            if (state_)
                throw_state_ = *state_;

            std::string const* auxinfo_ = xi.get<pika::detail::throw_auxinfo>();
            if (auxinfo_)
                throw_auxinfo_ = *auxinfo_;
        }
        catch (...)
        {
            // do nothing
        }

        // figure out concrete underlying exception type
        try
        {
            std::rethrow_exception(ep);
        }
        catch (pika::thread_interrupted const&)
        {
            type = pika::util::pika_thread_interrupted_exception;
            what = "pika::thread_interrupted";
            err_value = pika::thread_cancelled;
        }
        catch (pika::exception const& e)
        {
            type = pika::util::pika_exception;
            what = e.what();
            err_value = e.get_error();
        }
        catch (std::system_error const& e)
        {
            type = pika::util::std_system_error;
            what = e.what();
            err_value = e.code().value();
            err_message = e.code().message();
        }
        catch (std::runtime_error const& e)
        {
            type = pika::util::std_runtime_error;
            what = e.what();
        }
        catch (std::invalid_argument const& e)
        {
            type = pika::util::std_invalid_argument;
            what = e.what();
        }
        catch (std::out_of_range const& e)
        {
            type = pika::util::std_out_of_range;
            what = e.what();
        }
        catch (std::logic_error const& e)
        {
            type = pika::util::std_logic_error;
            what = e.what();
        }
        catch (std::bad_alloc const& e)
        {
            type = pika::util::std_bad_alloc;
            what = e.what();
        }
        catch (std::bad_cast const& e)
        {
            type = pika::util::std_bad_cast;
            what = e.what();
        }
        catch (std::bad_typeid const& e)
        {
            type = pika::util::std_bad_typeid;
            what = e.what();
        }
        catch (std::bad_exception const& e)
        {
            type = pika::util::std_bad_exception;
            what = e.what();
        }
        catch (std::exception const& e)
        {
            type = pika::util::std_exception;
            what = e.what();
        }
#if ASIO_HAS_BOOST_THROW_EXCEPTION != 0
        catch (boost::exception const& e)
        {
            type = pika::util::boost_exception;
            what = boost::diagnostic_information(e);
        }
#endif
        catch (...)
        {
            type = pika::util::unknown_exception;
            what = "unknown exception";
        }

        // clang-format off
        ar & type & what & throw_function_ & throw_file_ & throw_line_ &
            throw_locality_ & throw_hostname_ & throw_pid_ & throw_shepherd_ &
            throw_thread_id_ & throw_thread_name_ & throw_back_trace_ &
            throw_env_ & throw_config_ & throw_state_ & throw_auxinfo_;
        // clang-format on

        if (pika::util::pika_exception == type)
        {
            // clang-format off
            ar & err_value;
            // clang-format on
        }
        else if (pika::util::std_system_error == type)
        {
            // clang-format off
            ar & err_value & err_message;
            // clang-format on
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // TODO: This is not scalable, and painful to update.
    void load_custom_exception(pika::serialization::input_archive& ar,
        std::exception_ptr& e, unsigned int)
    {
        pika::util::exception_type type(pika::util::unknown_exception);
        std::string what;
        int err_value = pika::success;
        std::string err_message;

        std::uint32_t throw_locality_ = 0;
        std::string throw_hostname_;
        std::int64_t throw_pid_ = -1;
        std::size_t throw_shepherd_ = 0;
        std::size_t throw_thread_id_ = 0;
        std::string throw_thread_name_;
        std::string throw_function_;
        std::string throw_file_;
        std::string throw_back_trace_;
        int throw_line_ = 0;
        std::string throw_env_;
        std::string throw_config_;
        std::string throw_state_;
        std::string throw_auxinfo_;

        // clang-format off
        ar & type & what & throw_function_ & throw_file_ & throw_line_ &
            throw_locality_ & throw_hostname_ & throw_pid_ & throw_shepherd_ &
            throw_thread_id_ & throw_thread_name_ & throw_back_trace_ &
            throw_env_ & throw_config_ & throw_state_ & throw_auxinfo_;
        // clang-format on

        if (pika::util::pika_exception == type)
        {
            // clang-format off
            ar & err_value;
            // clang-format on
        }
        else if (pika::util::std_system_error == type)
        {
            // clang-format off
            ar & err_value & err_message;
            // clang-format on
        }

        switch (type)
        {
        default:
        case pika::util::std_exception:
        case pika::util::unknown_exception:
            e = pika::detail::construct_exception(
                pika::detail::std_exception(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        // standard exceptions
        case pika::util::std_runtime_error:
            e = pika::detail::construct_exception(std::runtime_error(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        case pika::util::std_invalid_argument:
            e = pika::detail::construct_exception(std::invalid_argument(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        case pika::util::std_out_of_range:
            e = pika::detail::construct_exception(std::out_of_range(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        case pika::util::std_logic_error:
            e = pika::detail::construct_exception(std::logic_error(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        case pika::util::std_bad_alloc:
            e = pika::detail::construct_exception(pika::detail::bad_alloc(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        case pika::util::std_bad_cast:
            e = pika::detail::construct_exception(pika::detail::bad_cast(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        case pika::util::std_bad_typeid:
            e = pika::detail::construct_exception(pika::detail::bad_typeid(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;
        case pika::util::std_bad_exception:
            e = pika::detail::construct_exception(
                pika::detail::bad_exception(what),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

#if ASIO_HAS_BOOST_THROW_EXCEPTION != 0
        // boost exceptions
        case pika::util::boost_exception:
            PIKA_ASSERT(false);    // shouldn't happen
            break;
#endif

        // boost::system::system_error
        case pika::util::boost_system_error:
            PIKA_FALLTHROUGH;

        // std::system_error
        case pika::util::std_system_error:
            e = pika::detail::construct_exception(
                std::system_error(
                    err_value, std::system_category(), err_message),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        // pika::exception
        case pika::util::pika_exception:
            e = pika::detail::construct_exception(
                pika::exception(
                    static_cast<pika::error>(err_value), what, pika::rethrow),
                pika::detail::construct_exception_info(throw_function_,
                    throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_));
            break;

        // pika::thread_interrupted
        case pika::util::pika_thread_interrupted_exception:
            e = pika::detail::construct_lightweight_exception(
                pika::thread_interrupted());
            break;
        }
    }
}}}    // namespace pika::runtime_local::detail
