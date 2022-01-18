//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

/// See N4310: 1.3/3
#include <algorithm>

// Parallelism TS V1
#include <pika/parallel/algorithms/adjacent_difference.hpp>
#include <pika/parallel/algorithms/adjacent_find.hpp>
#include <pika/parallel/algorithms/all_any_none.hpp>
#include <pika/parallel/algorithms/copy.hpp>
#include <pika/parallel/algorithms/count.hpp>
#include <pika/parallel/algorithms/equal.hpp>
#include <pika/parallel/algorithms/fill.hpp>
#include <pika/parallel/algorithms/find.hpp>
#include <pika/parallel/algorithms/for_each.hpp>
#include <pika/parallel/algorithms/generate.hpp>
#include <pika/parallel/algorithms/includes.hpp>
#include <pika/parallel/algorithms/is_heap.hpp>
#include <pika/parallel/algorithms/is_partitioned.hpp>
#include <pika/parallel/algorithms/is_sorted.hpp>
#include <pika/parallel/algorithms/lexicographical_compare.hpp>
#include <pika/parallel/algorithms/make_heap.hpp>
#include <pika/parallel/algorithms/merge.hpp>
#include <pika/parallel/algorithms/minmax.hpp>
#include <pika/parallel/algorithms/mismatch.hpp>
#include <pika/parallel/algorithms/move.hpp>
#include <pika/parallel/algorithms/nth_element.hpp>
#include <pika/parallel/algorithms/partial_sort.hpp>
#include <pika/parallel/algorithms/partial_sort_copy.hpp>
#include <pika/parallel/algorithms/partition.hpp>
#include <pika/parallel/algorithms/remove.hpp>
#include <pika/parallel/algorithms/remove_copy.hpp>
#include <pika/parallel/algorithms/replace.hpp>
#include <pika/parallel/algorithms/reverse.hpp>
#include <pika/parallel/algorithms/rotate.hpp>
#include <pika/parallel/algorithms/search.hpp>
#include <pika/parallel/algorithms/set_difference.hpp>
#include <pika/parallel/algorithms/set_intersection.hpp>
#include <pika/parallel/algorithms/set_symmetric_difference.hpp>
#include <pika/parallel/algorithms/set_union.hpp>
#include <pika/parallel/algorithms/sort.hpp>
#include <pika/parallel/algorithms/stable_sort.hpp>
#include <pika/parallel/algorithms/swap_ranges.hpp>
#include <pika/parallel/algorithms/unique.hpp>

// Parallelism TS V2
#include <pika/parallel/algorithms/ends_with.hpp>
#include <pika/parallel/algorithms/for_loop.hpp>
#include <pika/parallel/algorithms/shift_left.hpp>
#include <pika/parallel/algorithms/shift_right.hpp>
#include <pika/parallel/algorithms/starts_with.hpp>
