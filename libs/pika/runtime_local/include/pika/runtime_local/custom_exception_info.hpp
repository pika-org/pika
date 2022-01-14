//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/functional/function.hpp>
#include <pika/modules/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <utility>

#include <pika/local/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace pika {
    /// \cond NODETAIL
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // Stores the information about the locality id the exception has been
        // raised on. This information will show up in error messages under the
        // [locality] tag.
        PIKA_DEFINE_ERROR_INFO(throw_locality, std::uint32_t);

        // Stores the information about the hostname of the locality the exception
        // has been raised on. This information will show up in error messages
        // under the [hostname] tag.
        PIKA_DEFINE_ERROR_INFO(throw_hostname, std::string);

        // Stores the information about the pid of the OS process the exception
        // has been raised on. This information will show up in error messages
        // under the [pid] tag.
        PIKA_DEFINE_ERROR_INFO(throw_pid, std::int64_t);

        // Stores the information about the shepherd thread the exception has been
        // raised on. This information will show up in error messages under the
        // [shepherd] tag.
        PIKA_DEFINE_ERROR_INFO(throw_shepherd, std::size_t);

        // Stores the information about the pika thread the exception has been
        // raised on. This information will show up in error messages under the
        // [thread_id] tag.
        PIKA_DEFINE_ERROR_INFO(throw_thread_id, std::size_t);

        // Stores the information about the pika thread name the exception has been
        // raised on. This information will show up in error messages under the
        // [thread_name] tag.
        PIKA_DEFINE_ERROR_INFO(throw_thread_name, std::string);

        // Stores the information about the stack backtrace at the point the
        // exception has been raised at. This information will show up in error
        // messages under the [stack_trace] tag.
        PIKA_DEFINE_ERROR_INFO(throw_stacktrace, std::string);

        // Stores the full execution environment of the locality the exception
        // has been raised in. This information will show up in error messages
        // under the [env] tag.
        PIKA_DEFINE_ERROR_INFO(throw_env, std::string);

        // Stores the full pika configuration information of the locality the
        // exception has been raised in. This information will show up in error
        // messages under the [config] tag.
        PIKA_DEFINE_ERROR_INFO(throw_config, std::string);

        // Stores the current runtime state. This information will show up in
        // error messages under the [state] tag.
        PIKA_DEFINE_ERROR_INFO(throw_state, std::string);

        // Stores additional auxiliary information (such as information about
        // the current parcel). This information will show up in error messages
        // under the [auxinfo] tag.
        PIKA_DEFINE_ERROR_INFO(throw_auxinfo, std::string);

        // Portably extract the current execution environment
        PIKA_EXPORT std::string get_execution_environment();

        // Report an early or late exception and locally abort execution. There
        // isn't anything more we could do.
        PIKA_NORETURN PIKA_EXPORT void report_exception_and_terminate(
            std::exception const&);
        PIKA_NORETURN PIKA_EXPORT void report_exception_and_terminate(
            std::exception_ptr const&);
        PIKA_NORETURN PIKA_EXPORT void report_exception_and_terminate(
            pika::exception const&);

        // Report an early or late exception and locally exit execution. There
        // isn't anything more we could do. The exception will be re-thrown
        // from pika::init
        PIKA_EXPORT void report_exception_and_continue(
            std::exception const&);
        PIKA_EXPORT void report_exception_and_continue(
            std::exception_ptr const&);
        PIKA_EXPORT void report_exception_and_continue(
            pika::exception const&);

        PIKA_EXPORT pika::exception_info construct_exception_info(
            std::string const& func, std::string const& file, long line,
            std::string const& back_trace, std::uint32_t node,
            std::string const& hostname, std::int64_t pid, std::size_t shepherd,
            std::size_t thread_id, std::string const& thread_name,
            std::string const& env, std::string const& config,
            std::string const& state_name, std::string const& auxinfo);

        template <typename Exception>
        PIKA_EXPORT std::exception_ptr construct_exception(
            Exception const& e, pika::exception_info info);

        PIKA_EXPORT void pre_exception_handler();

        using get_full_build_string_type =
            pika::util::function_nonser<std::string()>;
        PIKA_EXPORT void set_get_full_build_string(
            get_full_build_string_type f);
        PIKA_EXPORT std::string get_full_build_string();

        PIKA_EXPORT std::string trace_on_new_stack(
            std::size_t frames_no = PIKA_HAVE_THREAD_BACKTRACE_DEPTH);
    }    // namespace detail

    namespace local::detail {
        PIKA_EXPORT pika::exception_info custom_exception_info(
            std::string const& func, std::string const& file, long line,
            std::string const& auxinfo);
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Extract the diagnostic information embedded in the given
    /// exception and return a string holding a formatted message.
    ///
    /// The function \a pika::diagnostic_information can be used to extract all
    /// diagnostic information stored in the given exception instance as a
    /// formatted string. This simplifies debug output as it composes the
    /// diagnostics into one, easy to use function call. This includes
    /// the name of the source file and line number, the sequence number of the
    /// OS-thread and the pika-thread id, the locality id and the stack backtrace
    /// of the point where the original exception was thrown.
    ///
    /// \param xi   The parameter \p e will be inspected for all diagnostic
    ///             information elements which have been stored at the point
    ///             where the exception was thrown. This parameter can be one
    ///             of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \returns    The formatted string holding all of the available
    ///             diagnostic information stored in the given exception
    ///             instance.
    ///
    /// \throws     std#bad_alloc (if any of the required allocation operations
    ///             fail)
    ///
    /// \see        \a pika::get_error_locality_id(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_thread_description(), \a pika::get_error(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::string diagnostic_information(
        exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string diagnostic_information(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? diagnostic_information(*xi) : std::string("<unknown>");
        });
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // Extract elements of the diagnostic information embedded in the given
    // exception.

    /// \brief Return the locality id where the exception was thrown.
    ///
    /// The function \a pika::get_error_locality_id can be used to extract the
    /// diagnostic information element representing the locality id as stored
    /// in the given exception instance.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \returns    The locality id of the locality where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return
    ///             \a pika::naming#invalid_locality_id.
    ///
    /// \throws     nothing
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_thread_description(), \a pika::get_error(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::uint32_t get_error_locality_id(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::uint32_t get_error_locality_id(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_locality_id(*xi) :
                        ~static_cast<std::uint32_t>(0);
        });
    }
    /// \endcond

    /// \brief Return the hostname of the locality where the exception was
    ///        thrown.
    ///
    /// The function \a pika::get_error_host_name can be used to extract the
    /// diagnostic information element representing the host name as stored in
    /// the given exception instance.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \returns    The hostname of the locality where the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return and empty string.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a pika::diagnostic_information()
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_thread_description(), \a pika::get_error()
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::string get_error_host_name(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_host_name(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_host_name(*xi) : std::string();
        });
    }
    /// \endcond

    /// \brief Return the (operating system) process id of the locality where
    ///        the exception was thrown.
    ///
    /// The function \a pika::get_error_process_id can be used to extract the
    /// diagnostic information element representing the process id as stored in
    /// the given exception instance.
    ///
    /// \returns    The process id of the OS-process which threw the exception
    ///             If the exception instance does not hold
    ///             this information, the function will return 0.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     nothing
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_thread_description(), \a pika::get_error(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::int64_t get_error_process_id(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::int64_t get_error_process_id(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_process_id(*xi) : -1;
        });
    }
    /// \endcond

    /// \brief Return the environment of the OS-process at the point the
    ///        exception was thrown.
    ///
    /// The function \a pika::get_error_env can be used to extract the
    /// diagnostic information element representing the environment of the
    /// OS-process collected at the point the exception was thrown.
    ///
    /// \returns    The environment from the point the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_thread_description(), \a pika::get_error(),
    ///             \a pika::get_error_backtrace(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::string get_error_env(pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_env(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_env(*xi) : std::string("<unknown>");
        });
    }
    /// \endcond

    /// \brief Return the stack backtrace from the point the exception was thrown.
    ///
    /// The function \a pika::get_error_backtrace can be used to extract the
    /// diagnostic information element representing the stack backtrace
    /// collected at the point the exception was thrown.
    ///
    /// \returns    The stack back trace from the point the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_thread_description(), \a pika::get_error(),
    ///             \a pika::get_error_env(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::string get_error_backtrace(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_backtrace(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_backtrace(*xi) : std::string();
        });
    }
    /// \endcond

    /// \brief Return the sequence number of the OS-thread used to execute
    ///        pika-threads from which the exception was thrown.
    ///
    /// The function \a pika::get_error_os_thread can be used to extract the
    /// diagnostic information element representing the sequence number  of the
    /// OS-thread as stored in the given exception instance.
    ///
    /// \returns    The sequence number of the OS-thread used to execute the
    ///             pika-thread from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return std::size(-1).
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     nothing
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_thread_id(),
    ///             \a pika::get_error_thread_description(), \a pika::get_error(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::size_t get_error_os_thread(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::size_t get_error_os_thread(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_os_thread(*xi) : std::size_t(-1);
        });
    }
    /// \endcond

    /// \brief Return the unique thread id of the pika-thread from which the
    ///        exception was thrown.
    ///
    /// The function \a pika::get_error_thread_id can be used to extract the
    /// diagnostic information element representing the pika-thread id
    /// as stored in the given exception instance.
    ///
    /// \returns    The unique thread id of the pika-thread from which the
    ///             exception was thrown. If the exception instance
    ///             does not hold this information, the function will return
    ///             std::size_t(0).
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     nothing
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread()
    ///             \a pika::get_error_thread_description(), \a pika::get_error(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error_what(), \a pika::get_error_config(),
    ///             \a pika::get_error_state()
    ///
    PIKA_EXPORT std::size_t get_error_thread_id(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::size_t get_error_thread_id(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_thread_id(*xi) : std::size_t(-1);
        });
    }
    /// \endcond

    /// \brief Return any additionally available thread description of the
    ///        pika-thread from which the exception was thrown.
    ///
    /// The function \a pika::get_error_thread_description can be used to extract the
    /// diagnostic information element representing the additional thread
    /// description as stored in the given exception instance.
    ///
    /// \returns    Any additionally available thread description of the
    ///             pika-thread from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error(), \a pika::get_error_state(),
    ///             \a pika::get_error_what(), \a pika::get_error_config()
    ///
    PIKA_EXPORT std::string get_error_thread_description(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_thread_description(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_thread_description(*xi) : std::string();
        });
    }
    /// \endcond

    /// \brief Return the pika configuration information point from which the
    ///        exception was thrown.
    ///
    /// The function \a pika::get_error_config can be used to extract the
    /// pika configuration information element representing the full pika
    /// configuration information as stored in the given exception instance.
    ///
    /// \returns    Any additionally available pika configuration information
    ///             the point from which the exception was
    ///             thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error(), \a pika::get_error_state()
    ///             \a pika::get_error_what(), \a pika::get_error_thread_description()
    ///
    PIKA_EXPORT std::string get_error_config(
        pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_config(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_config(*xi) : std::string();
        });
    }
    /// \endcond

    /// \brief Return the pika runtime state information at which the exception
    ///        was thrown.
    ///
    /// The function \a pika::get_error_state can be used to extract the
    /// pika runtime state information element representing the state the
    /// runtime system is currently in as stored in the given exception
    /// instance.
    ///
    /// \returns    The point runtime state at the point at which the exception
    ///             was thrown. If the exception instance does not hold
    ///             this information, the function will return an empty string.
    ///
    /// \param xi   The parameter \p e will be inspected for the requested
    ///             diagnostic information elements which have been stored at
    ///             the point where the exception was thrown. This parameter
    ///             can be one of the following types: \a pika::exception_info,
    ///             \a pika::error_code, \a std::exception, or
    ///             \a std::exception_ptr.
    ///
    /// \throws     std#bad_alloc (if one of the required allocations fails)
    ///
    /// \see        \a pika::diagnostic_information(), \a pika::get_error_host_name(),
    ///             \a pika::get_error_process_id(), \a pika::get_error_function_name(),
    ///             \a pika::get_error_file_name(), \a pika::get_error_line_number(),
    ///             \a pika::get_error_os_thread(), \a pika::get_error_thread_id(),
    ///             \a pika::get_error_backtrace(), \a pika::get_error_env(),
    ///             \a pika::get_error(),
    ///             \a pika::get_error_what(), \a pika::get_error_thread_description()
    ///
    PIKA_EXPORT std::string get_error_state(pika::exception_info const& xi);

    /// \cond NOINTERNAL
    template <typename E>
    std::string get_error_state(E const& e)
    {
        return invoke_with_exception_info(e, [](exception_info const* xi) {
            return xi ? get_error_state(*xi) : std::string();
        });
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // \cond NOINTERNAL
    // For testing purposes we sometime expect to see exceptions, allow those
    // to go through without attaching a debugger.
    //
    // This should be used carefully as it disables the possible attaching of
    // a debugger for all exceptions, not only the expected ones.
    PIKA_EXPORT bool expect_exception(bool flag = true);
    /// \endcond

}    // namespace pika

#include <pika/modules/errors.hpp>

#include <pika/local/config/warnings_suffix.hpp>
