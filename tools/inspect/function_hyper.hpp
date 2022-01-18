//  Hyperlink Function  ------------------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/modules/filesystem.hpp>

#include "inspector.hpp"

#include <string>

using pika::filesystem::path;

// When you have a specific line and the line is the location of the link
inline std::string linelink(path const& full_path, std::string const& linenumb)
{
    std::string commit = PIKA_HAVE_GIT_COMMIT;
    std::string location = boost::inspect::relative_to(
        full_path, boost::inspect::search_root_path() );
    //The erase function for location is to get rid of the first /pika that will always
    //be present in any full path this tool is used for (repeated in wordlink and
    // loclink)
    std::string total = "<a href = \"https://github.com/pika-org/pika/blob/"
        + commit + location + "#L" + linenumb + "\">";
    total = total + linenumb;
    total = total + "</a>";
    return total;
}

// When you have a specific line, but a word is the location of the link
inline std::string wordlink(
    path const& full_path, std::string const& linenumb, std::string const& word)
{
    std::string commit = PIKA_HAVE_GIT_COMMIT;
    std::string location = boost::inspect::relative_to(
        full_path, boost::inspect::search_root_path() );
    std::string total = "<a href = \"https://github.com/pika-org/pika/blob/"
        + commit + location + "#L" + linenumb + "\">";
    total = total + word;
    total = total + "</a>";
    return total;
}

// When you don't have a specific line
inline std::string loclink(path const& full_path, std::string const& word)
{
    std::string commit = PIKA_HAVE_GIT_COMMIT;
    std::string location = boost::inspect::relative_to(
        full_path, boost::inspect::search_root_path() );
    std::string total = "<a href = \"https://github.com/pika-org/pika/blob/"
        + commit + location + "\">";
    total = total + word;
    total = total + "</a>";
    return total;
}

