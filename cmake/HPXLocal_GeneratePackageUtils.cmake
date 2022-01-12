# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Replace the "-NOTFOUND" by empty var in case the property is not found
macro(hpx_local_get_target_property var target property)
  get_target_property(${var} ${target} ${property})
  list(FILTER ${var} EXCLUDE REGEX "-NOTFOUND$")
endmacro(hpx_local_get_target_property)
