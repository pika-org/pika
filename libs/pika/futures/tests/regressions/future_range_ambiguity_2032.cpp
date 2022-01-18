//  Copyright (c) 2016 Denis Demidov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This tests verifies that #2032 remains fixed

#include <pika/iterator_support/iterator_range.hpp>
#include <pika/local/future.hpp>

#include <vector>

typedef pika::util::iterator_range<
    std::vector<pika::shared_future<void>>::iterator>
    future_range;

typedef pika::traits::is_future_range<future_range>::type error1;

typedef pika::traits::future_range_traits<future_range>::future_type error2;

int main()
{
    return 0;
}
