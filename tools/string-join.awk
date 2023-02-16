# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# join strings across lines, not foolproof but handles 99% of c++ cases
# can fail for empty strings like
# std::string x = "stuff " + "    "
#                 " more string carried over from above"
#

function clear_buffer() {
    if (buffer) print buffer
    buffer=""
    in_string=0
}

# remove first instance of "   "
function join_strings(buf) {
    sub("\"[[:space:]]*\"", "", buf)
    buffer=buf
}

BEGIN {
    enabled=1
    in_string=0 
}

# if we see "clang-format off" turn off processing
# if we see "clang-forma on" turn back on processing
/\/\/ clang-format off/    { enabled=0; clear_buffer(); print $0; next }
/\/\/ clang-format on/     { enabled=1; print $0; next }

{ 
    string_start=0
    string_end=0
    if (!enabled) {
        print $0
        next
    }
}

# line starts with a string quote (ignoring leading whitespace)
/^[[:space:]]*\"/    { string_start=1 }
# line ends with a string quote (ignoring trailing whitespace)
/\"[[:space:]]*$/    { string_end=1 }
# disallow string ending with \n"
/\\n\"[[:space:]]*$/ { string_end=0 }
# disallow string ending with \"
/\\"[[:space:]]*$/   { string_end=0 ; print "2"}

# if we're in a string and this line starts with string : join
in_string && string_start {
    buffer=buffer $0
    join_strings(buffer)
    if (!string_end)
        clear_buffer()
    next
}

# if current line ends on a string quote, then start new buffer
!in_string && string_end { 
    buffer=$0
    in_string=1
    next
}

# if we were in a string, terminate
in_string {
    clear_buffer()
}

# default fallthrough, just print the current line
{
    print $0
}

END {
    if (buffer) print buffer
}
