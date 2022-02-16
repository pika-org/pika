# Copyright (c) 2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add inpect executable

include(pika_setup_boost_regex)

pika_add_executable(
  inspect INTERNAL_FLAGS AUTOGLOB NOLIBS FOLDER "Tools/Inspect"
)

if(NOT Boost_REGEX_FOUND)
  pika_error("pika inspect tool requires Boost.Regex")
endif()

# Set the basic search paths for the generated pika headers
target_link_libraries(inspect PRIVATE Boost::regex pika_dependencies_boost pika)

# add dependencies to pseudo-target
pika_add_pseudo_dependencies(tools.inspect inspect)
