////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>
#include <pika/assert.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/modules/errors.hpp>
#include <pika/prefix/find_prefix.hpp>
#include <pika/string_util/classification.hpp>
#include <pika/string_util/split.hpp>
#include <pika/type_support/unused.hpp>

#if defined(PIKA_WINDOWS)
# include <windows.h>
#elif defined(__linux) || defined(linux) || defined(__linux__)
# include <linux/limits.h>
# include <sys/stat.h>
# include <unistd.h>
# include <vector>
#elif __APPLE__
# include <mach-o/dyld.h>
#elif defined(__FreeBSD__)
# include <algorithm>
# include <iterator>
# include <sys/sysctl.h>
# include <sys/types.h>
# include <vector>
#endif

#include <boost/tokenizer.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.prefix;
#endif

namespace pika::util {
    ///////////////////////////////////////////////////////////////////////////////
    std::string get_executable_prefix(char const* argv0)
    {
        std::filesystem::path p(get_executable_filename(argv0));

        return p.parent_path().parent_path().string();
    }

    std::string get_executable_filename(char const* argv0)
    {
        std::string r;

// #if defined(PIKA_WINDOWS)
//         PIKA_UNUSED(argv0);

//         char exe_path[MAX_PATH + 1] = {'\0'};
//         if (!GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path)))
//         {
//             PIKA_THROW_EXCEPTION(pika::error::dynamic_link_failure, "get_executable_filename",
//                 "unable to find executable filename");
//         }
//         r = exe_path;

// #elif defined(__linux) || defined(linux) || defined(__linux__)
//         char buf[PATH_MAX + 1];
//         ssize_t length = ::readlink("/proc/self/exe", buf, sizeof(buf));

//         if (length != -1)
//         {
//             buf[length] = '\0';
//             r = buf;
//             return r;
//         }

//         std::string argv0_(argv0);

//         // REVIEW: Should we resolve symlinks at any point here?
//         if (argv0_.length() > 0)
//         {
//             // Check for an absolute path.
//             if (argv0_[0] == '/') return argv0_;

//             // Check for a relative path.
//             if (argv0_.find('/') != std::string::npos)
//             {
//                 // Get the current working directory.

//                 // NOTE: getcwd does give you a null terminated string,
//                 // while readlink (above) does not.
//                 if (::getcwd(buf, PATH_MAX))
//                 {
//                     r = buf;
//                     r += '/';
//                     r += argv0_;
//                     return r;
//                 }
//             }

//             // Search PATH
//             char const* epath = ::getenv("PATH");
//             if (epath)
//             {
//                 std::vector<std::string> path_dirs;

//                 pika::detail::split(path_dirs, epath, pika::detail::is_any_of(":"),
//                     pika::detail::token_compress_mode::on);

//                 for (std::uint64_t i = 0; i < path_dirs.size(); ++i)
//                 {
//                     r = path_dirs[i];
//                     r += '/';
//                     r += argv0_;

//                     // Can't use Boost.Filesystem as it doesn't let me access
//                     // st_uid and st_gid.
//                     struct stat s;

//                     // Make sure the file is executable and shares our
//                     // effective uid and gid.
//                     // NOTE: If someone was using a pika application that was
//                     // seteuid'd to root, this may fail.
//                     if (0 == ::stat(r.c_str(), &s))
//                         if ((s.st_uid == ::geteuid()) && (s.st_mode & S_IXUSR) &&
//                             (s.st_gid == ::getegid()) && (s.st_mode & S_IXGRP) &&
//                             (s.st_mode & S_IXOTH))
//                             return r;
//                 }
//             }
//         }

//         PIKA_THROW_EXCEPTION(pika::error::dynamic_link_failure, "get_executable_filename",
//             "unable to find executable filename");

// #elif defined(__APPLE__)
//         PIKA_UNUSED(argv0);

//         char exe_path[PATH_MAX + 1];
//         std::uint32_t len = sizeof(exe_path) / sizeof(exe_path[0]);

//         if (0 != _NSGetExecutablePath(exe_path, &len))
//         {
//             PIKA_THROW_EXCEPTION(pika::error::dynamic_link_failure, "get_executable_filename",
//                 "unable to find executable filename");
//         }
//         exe_path[len - 1] = '\0';
//         r = exe_path;

// #elif defined(__FreeBSD__)
//         PIKA_UNUSED(argv0);

//         int mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
//         size_t cb = 0;
//         sysctl(mib, 4, nullptr, &cb, nullptr, 0);
//         if (cb)
//         {
//             std::vector<char> buf(cb);
//             sysctl(mib, 4, &buf[0], &cb, nullptr, 0);
//             std::copy(buf.begin(), buf.end(), std::back_inserter(r));
//         }

// #else
// # error Unsupported platform
// #endif

        return r;
    }
}    // namespace pika::util
