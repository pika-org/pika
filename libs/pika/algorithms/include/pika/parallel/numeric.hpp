//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

/// See N4310: 1.3/3
#include <numeric>

#include <pika/parallel/algorithms/adjacent_difference.hpp>
#include <pika/parallel/algorithms/exclusive_scan.hpp>
#include <pika/parallel/algorithms/inclusive_scan.hpp>
#include <pika/parallel/algorithms/reduce.hpp>
#include <pika/parallel/algorithms/transform_exclusive_scan.hpp>
#include <pika/parallel/algorithms/transform_inclusive_scan.hpp>
#include <pika/parallel/algorithms/transform_reduce.hpp>
