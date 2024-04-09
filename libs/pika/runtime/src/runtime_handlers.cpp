//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/debugging/backtrace.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/logging.hpp>
#include <pika/modules/thread_manager.hpp>
#include <pika/runtime/config_entry.hpp>
#include <pika/runtime/custom_exception_info.hpp>
#include <pika/runtime/debugging.hpp>
#include <pika/runtime/runtime.hpp>
#include <pika/runtime/runtime_handlers.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

namespace pika::detail {

    [[noreturn]] void assertion_handler(
        pika::detail::source_location const& loc, const char* expr, std::string const& msg)
    {
        static thread_local bool handling_assertion = false;

        if (handling_assertion)
        {
            std::ostringstream strm;
            strm << "Trying to handle failed assertion while handling another failed assertion!"
                 << std::endl;
            strm << "Assertion '" << expr << "' failed";
            if (!msg.empty()) { strm << " (" << msg << ")"; }

            strm << std::endl;
            strm << "{file}: " << loc.file_name << std::endl;
            strm << "{line}: " << loc.line_number << std::endl;
            strm << "{function}: " << loc.function_name << std::endl;

            std::cerr << strm.str();

            std::abort();
        }

        handling_assertion = true;

        std::ostringstream strm;
        strm << "Assertion '" << expr << "' failed";
        if (!msg.empty()) { strm << " (" << msg << ")"; }

        pika::exception e(pika::error::assertion_failure, strm.str());
        std::cerr << pika::diagnostic_information(pika::detail::get_exception(
                         e, loc.function_name, loc.file_name, loc.line_number))
                  << std::endl;

        pika::util::may_attach_debugger("exception");

        std::abort();
    }

#if defined(PIKA_HAVE_VERIFY_LOCKS)
    void registered_locks_error_handler()
    {
        std::string back_trace = pika::debug::detail::trace(std::size_t(128));

        // throw or log, depending on config options
        if (get_config_entry("pika.throw_on_held_lock", "1") == "0")
        {
            if (back_trace.empty())
            {
                PIKA_LOG(debug,
                    "suspending thread while at least one lock is being held "
                    "(stack backtrace was disabled at compile time)");
            }
            else
            {
                PIKA_LOG(debug,
                    "suspending thread while at least one lock is being held, stack backtrace: {}",
                    back_trace);
            }
        }
        else
        {
            if (back_trace.empty())
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status, "verify_no_locks",
                    "suspending thread while at least one lock is being held (stack backtrace was "
                    "disabled at compile time)");
            }
            else
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status, "verify_no_locks",
                    "suspending thread while at least one lock is being held, stack backtrace: {}",
                    back_trace);
            }
        }
    }

    bool register_locks_predicate() { return threads::detail::get_self_ptr() != nullptr; }
#endif

    threads::detail::thread_pool_base* get_default_pool()
    {
        pika::runtime* rt = get_runtime_ptr();
        if (rt == nullptr)
        {
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::detail::get_default_pool",
                "The runtime system is not active");
        }

        return &rt->get_thread_manager().default_pool();
    }

    threads::detail::mask_cref_type get_pu_mask(
        threads::detail::topology& /* topo */, std::size_t thread_num)
    {
        auto& rp = pika::resource::get_partitioner();
        return rp.get_pu_mask(thread_num);
    }
}    // namespace pika::detail
