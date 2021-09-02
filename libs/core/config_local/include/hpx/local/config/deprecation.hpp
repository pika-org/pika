//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config/defines.hpp>
#include <hpx/local/config/attributes.hpp>
#include <hpx/local/config/version.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting HPXLocal V0.1
#if !defined(HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V0_1)
#define HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V0_1 1
#endif

#if (HPX_LOCAL_VERSION_FULL >= 0x000100) &&                                    \
    (HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V0_1 != 0)
#define HPX_LOCAL_DEPRECATED_MSG_V0_1                                          \
    "This functionality is deprecated starting HPXLocal V0.1 and will be "     \
    "removed "                                                                 \
    "in the future. You can define "                                           \
    "HPX_LOCAL_HAVE_DEPRECATION_WARNINGS_V0_1=0 to "                           \
    "acknowledge that you have received this warning."
#define HPX_LOCAL_DEPRECATED_V0_1(x)                                           \
    HPX_LOCAL_DEPRECATED(                                                      \
        x " (" HPX_PP_EXPAND(HPX_LOCAL_DEPRECATED_MSG_V0_1) ")")
#endif

#if !defined(HPX_LOCAL_DEPRECATED_V0_1)
#define HPX_LOCAL_DEPRECATED_V0_1(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Deprecate a given functionality starting at the given version of HPX
#define HPX_LOCAL_DEPRECATED_V(major, minor, x)                                \
    HPX_PP_CAT(                                                                \
        HPX_PP_CAT(HPX_PP_CAT(HPX_LOCAL_DEPRECATED_V, major), _), minor)       \
    (x)
