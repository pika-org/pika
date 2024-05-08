//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/debugging/print.hpp>

namespace pika::cuda::experimental::detail {
    using namespace pika::debug::detail;
    template <int Level>
    static print_threshold<Level, 0> cud_debug("CUDA-EX");
}    // namespace pika::cuda::experimental::detail
