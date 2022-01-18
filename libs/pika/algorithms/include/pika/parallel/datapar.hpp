//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR)

#include <pika/executors/datapar/execution_policy.hpp>
#include <pika/parallel/datapar/adjacent_difference.hpp>
#include <pika/parallel/datapar/fill.hpp>
#include <pika/parallel/datapar/find.hpp>
#include <pika/parallel/datapar/generate.hpp>
#include <pika/parallel/datapar/iterator_helpers.hpp>
#include <pika/parallel/datapar/loop.hpp>
#include <pika/parallel/datapar/transfer.hpp>
#include <pika/parallel/datapar/transform_loop.hpp>
#include <pika/parallel/datapar/zip_iterator.hpp>

#endif
