//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(PIKA_HAVE_CXX23_STATIC_CALL_OPERATOR) &&                                               \
    (!defined(PIKA_HAVE_GPU_SUPPORT) || defined(PIKA_HAVE_CXX23_STATIC_CALL_OPERATOR_GPU))
# define PIKA_STATIC_CALL_OPERATOR(...) static operator()(__VA_ARGS__)
#else
# define PIKA_STATIC_CALL_OPERATOR(...) operator()(__VA_ARGS__) const
#endif
