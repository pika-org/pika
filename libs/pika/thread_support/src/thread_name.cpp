//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code was partially taken from:
//      http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx

#include <pika/config.hpp>
#include <pika/thread_support/thread_name.hpp>
#include <pika/type_support/unused.hpp>

#include <fmt/format.h>

#if defined(PIKA_HAVE_PTHREAD_SETNAME_NP)
# include <pthread.h>
#endif

#include <stdexcept>
#include <string>

namespace pika::detail {
    std::string& get_thread_name_internal()
    {
        static thread_local std::string name{};
        return name;
    }

    std::string get_thread_name()
    {
        std::string const& name = detail::get_thread_name_internal();
        if (name.empty()) return "<unknown>";
        return name;
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
    DWORD const MS_VC_EXCEPTION = 0x406D'1388;

# pragma pack(push, 8)
    typedef struct tagTHREADNAME_INFO
    {
        DWORD dwType;        // Must be 0x1000.
        LPCSTR szName;       // Pointer to name (in user addr space).
        DWORD dwThreadID;    // Thread ID (-1=caller thread).
        DWORD dwFlags;       // Reserved for future use, must be zero.
    } THREADNAME_INFO;
# pragma pack(pop)

    // Set the name of the thread shown in the Visual Studio debugger
    void set_thread_name(std::string_view name, std::string_view)
    {
        auto& name_internal = get_thread_name_internal();
        name_internal = name;

        DWORD dwThreadID = -1;
        THREADNAME_INFO info;
        info.dwType = 0x1000;
        info.szName = name_internal.c_str();
        info.dwThreadID = dwThreadID;
        info.dwFlags = 0;

        __try
        {
            RaiseException(
                MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*) &info);
        }
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
        }
    }
#else
    void set_thread_name(std::string_view name, std::string_view short_name)
    {
        auto& name_internal = get_thread_name_internal();
        name_internal = name;
# if defined(PIKA_HAVE_PTHREAD_SETNAME_NP)
        // https://man7.org/linux/man-pages/man3/pthread_setname_np.3.html:
        // The thread name is a meaningful C language string, whose length is restricted to 16
        // characters, including the terminating null byte ('\0').
        std::string pthread_name{short_name, 0, 15};
        int rc = pthread_setname_np(pthread_self(), pthread_name.c_str());
        if (rc != 0)
        {
            throw std::runtime_error(fmt::format("pthread_setname_np failed with code {} when "
                                                 "attempting to set name \"{}\" for thread {}",
                rc, pthread_name, pthread_self()));
        }
# else
        PIKA_UNUSED(short_name);
# endif
    }
#endif
}    // namespace pika::detail
