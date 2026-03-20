//  Copyright (c) 2026 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <exec/split.hpp>

// Newer versions of stdexec move execute and split to a different namespace. This test checks if
// the exec/split.hpp header is available.
int main() { using ::experimental::execution::split_t; }
