//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_TRACY)
# if defined(__has_include)
// Newer versions of tracy have Tracy.hpp in the subdirectory tracy
#  if __has_include(<tracy/Tracy.hpp>)
#   include <tracy/Tracy.hpp>
#  else
#   include <Tracy.hpp>
#  endif
// If we can't detect tracy's includes we assume it is new enough to use the
// tracy subdirectory
# else
#  include <tracy/Tracy.hpp>
# endif
#endif
