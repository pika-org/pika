//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if defined(__NVCOMPILER)
# pragma diag_error static_member_operator_not_allowed
#endif

struct s
{
    static void operator()() {}
};

int main() { s::operator()(); }

#if defined(__NVCOMPILER)
# pragma diag_default static_member_operator_not_allowed
#endif
