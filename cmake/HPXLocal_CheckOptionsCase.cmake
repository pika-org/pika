# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This file checks that all cache variables that start with HPXLocal use the
# correct casing. We also check for HPX_LOCAL.
get_cmake_property(cache_vars CACHE_VARIABLES)

foreach(var IN LISTS cache_vars)
  string(TOLOWER ${var} var_lc)
  if((var_lc MATCHES "^hpx_local") OR ((var_lc MATCHES "^hpxlocal")
                                       AND NOT (var MATCHES "^HPXLocal"))
  )
    list(APPEND inconsistent_vars ${var})
  endif()
endforeach()
if(inconsistent_vars)
  hpx_local_error(
    "HPXLocal expects all HPXLocal variables to be prefixed with HPXLocal. \
Found the following variables with inconsistent casing: ${inconsistent_vars}. \
Please unset the listed variables and set them with the correctly cased prefix \
instead."
  )
endif()
