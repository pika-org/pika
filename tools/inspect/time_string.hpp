// ---- time_string: thin wrapper around std::strftime    -------- //
//
//            Copyright Gennaro Prota 2006
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// ------------------------------------------------------------------
//
// $Id$

#pragma once

#include <ctime>
#include <string>

namespace boost {

    // Many of the boost tools just need a quick way to obtain
    // a formatted "run date" string or similar. This is one.
    //
    // In case of failure false is returned and result is
    // unchanged.
    //
    inline bool time_string(
        std::string& result, const std::string& format = "%X UTC, %A %d %B %Y")
    {
        // give up qualifying names and using std::size_t,
        // to avoid including "config.hpp"
        using namespace std;

        const int sz = 256;
        char buffer[sz] = {0};
        const time_t no_cal_time(-1);
        time_t tod;

        const bool ok = time(&tod) != no_cal_time &&
            strftime(buffer, sz, format.c_str(), gmtime(&tod)) != 0;

        if (ok)
            result = buffer;

        return ok;
    }

}    // namespace boost
