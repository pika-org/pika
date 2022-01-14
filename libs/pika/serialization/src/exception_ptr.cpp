//  Copyright (c)      2020 ETH Zurich
//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/modules/errors.hpp>
#include <pika/serialization/exception_ptr.hpp>
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
namespace pika { namespace serialization {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////////
        // TODO: This is not scalable, and painful to update.
        void save(output_archive& ar, std::exception_ptr const& ep,
            unsigned int /* version */)
        {
            pika::util::exception_type type(pika::util::unknown_exception);
            std::string what;
            int err_value = pika::success;
            std::string err_message;

            std::string throw_function_;
            std::string throw_file_;
            long throw_line_ = 0;

            // retrieve information related to exception_info
            try
            {
                std::rethrow_exception(ep);
            }
            catch (exception_info const& xi)
            {
                std::string const* function =
                    xi.get<pika::detail::throw_function>();
                if (function)
                    throw_function_ = *function;

                std::string const* file = xi.get<pika::detail::throw_file>();
                if (file)
                    throw_file_ = *file;

                long const* line = xi.get<pika::detail::throw_line>();
                if (line)
                    throw_line_ = *line;
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
#if BOOST_ASIO_HAS_BOOST_THROW_EXCEPTION != 0
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
            ar & type & what & throw_function_ & throw_file_ & throw_line_;
            // clang-format on

            if (pika::util::pika_exception == type)
            {
                // clang-format off
                ar & err_value;
                // clang-format on
            }
            else if (pika::util::boost_system_error == type ||
                pika::util::std_system_error == type)
            {
                // clang-format off
                ar & err_value & err_message;
                // clang-format on
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // TODO: This is not scalable, and painful to update.
        void load(input_archive& ar, std::exception_ptr& e,
            unsigned int /* version */)
        {
            pika::util::exception_type type(pika::util::unknown_exception);
            std::string what;
            int err_value = pika::success;
            std::string err_message;

            std::string throw_function_;
            std::string throw_file_;
            int throw_line_ = 0;

            // clang-format off
            ar & type & what & throw_function_ & throw_file_ & throw_line_;
            // clang-format on

            if (pika::util::pika_exception == type)
            {
                // clang-format off
                ar & err_value;
                // clang-format on
            }
            else if (pika::util::boost_system_error == type ||
                pika::util::std_system_error == type)
            {
                // clang-format off
                ar & err_value& err_message;
                // clang-format on
            }

            switch (type)
            {
            default:
            case pika::util::std_exception:
            case pika::util::unknown_exception:
                e = pika::detail::get_exception(pika::detail::std_exception(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            // standard exceptions
            case pika::util::std_runtime_error:
                e = pika::detail::get_exception(std::runtime_error(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case pika::util::std_invalid_argument:
                e = pika::detail::get_exception(std::invalid_argument(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case pika::util::std_out_of_range:
                e = pika::detail::get_exception(std::out_of_range(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case pika::util::std_logic_error:
                e = pika::detail::get_exception(std::logic_error(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case pika::util::std_bad_alloc:
                e = pika::detail::get_exception(pika::detail::bad_alloc(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case pika::util::std_bad_cast:
                e = pika::detail::get_exception(pika::detail::bad_cast(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case pika::util::std_bad_typeid:
                e = pika::detail::get_exception(pika::detail::bad_typeid(what),
                    throw_function_, throw_file_, throw_line_);
                break;
            case pika::util::std_bad_exception:
                e = pika::detail::get_exception(pika::detail::bad_exception(what),
                    throw_function_, throw_file_, throw_line_);
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
                e = pika::detail::get_exception(
                    std::system_error(
                        err_value, std::system_category(), err_message),
                    throw_function_, throw_file_, throw_line_);
                break;

            // pika::exception
            case pika::util::pika_exception:
                e = pika::detail::get_exception(
                    pika::exception(
                        static_cast<pika::error>(err_value), what, pika::rethrow),
                    throw_function_, throw_file_, throw_line_);
                break;

            // pika::thread_interrupted
            case pika::util::pika_thread_interrupted_exception:
                e = pika::detail::construct_lightweight_exception(
                    pika::thread_interrupted());
                break;
            }
        }

        save_custom_exception_handler_type& get_save_custom_exception_handler()
        {
            static save_custom_exception_handler_type f = save;
            return f;
        }

        PIKA_EXPORT void set_save_custom_exception_handler(
            save_custom_exception_handler_type f)
        {
            get_save_custom_exception_handler() = f;
        }

        load_custom_exception_handler_type& get_load_custom_exception_handler()
        {
            static load_custom_exception_handler_type f = load;
            return f;
        }

        PIKA_EXPORT void set_load_custom_exception_handler(
            load_custom_exception_handler_type f)
        {
            get_load_custom_exception_handler() = f;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void save(Archive& ar, std::exception_ptr const& ep, unsigned int version)
    {
        if (detail::get_save_custom_exception_handler())
        {
            detail::get_save_custom_exception_handler()(ar, ep, version);
        }
        else
        {
            PIKA_THROW_EXCEPTION(invalid_status, "pika::serialization::save",
                "Attempted to save a std::exception_ptr, but there is no "
                "handler installed. Set one with "
                "pika::serialization::detail::set_save_custom_exception_"
                "handler.");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void load(Archive& ar, std::exception_ptr& ep, unsigned int version)
    {
        if (detail::get_load_custom_exception_handler())
        {
            detail::get_load_custom_exception_handler()(ar, ep, version);
        }
        else
        {
            PIKA_THROW_EXCEPTION(invalid_status, "pika::serialization::load",
                "Attempted to load a std::exception_ptr, but there is no "
                "handler installed. Set one with "
                "pika::serialization::detail::set_load_custom_exception_"
                "handler.");
        }
    }

    template PIKA_EXPORT void save(pika::serialization::output_archive&,
        std::exception_ptr const&, unsigned int);

    template PIKA_EXPORT void load(
        pika::serialization::input_archive&, std::exception_ptr&, unsigned int);
}}    // namespace pika::serialization
