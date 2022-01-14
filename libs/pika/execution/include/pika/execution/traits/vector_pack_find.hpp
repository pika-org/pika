//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)

#if !defined(__CUDACC__)
#include <pika/execution/traits/detail/simd/vector_pack_find.hpp>
#include <pika/execution/traits/detail/vc/vector_pack_find.hpp>
#endif

#endif
