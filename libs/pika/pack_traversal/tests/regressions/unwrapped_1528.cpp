//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/futures/future.hpp>
#include <pika/pack_traversal/unwrap.hpp>

#include <vector>

void noop() {}

int main()
{
    std::vector<pika::future<void>> fs;
    pika::unwrapping (&noop)(fs);
}
