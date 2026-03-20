//  Copyright (c) 2026 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <exec/execute.hpp>

// Newer versions of stdexec move execute to a different namespace. This test checks if
// the exec/execute.hpp header is available.
int main() { using ::experimental::execution::__execute_t; }
