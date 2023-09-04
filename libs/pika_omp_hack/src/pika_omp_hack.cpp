//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/topology/topology.hpp>

namespace pika::detail {
    void preinit_get_topology() { pika::threads::detail::get_topology(); }

    // Based on the example in https://reviews.llvm.org/D20646?id=58513
    [[maybe_unused]] __attribute__((section(".preinit_array"), used)) void (*pika_preinit_array)(
        void) = preinit_get_topology;

}    // namespace pika::detail
