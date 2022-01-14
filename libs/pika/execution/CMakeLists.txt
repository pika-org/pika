# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(execution_headers
    pika/execution/algorithms/bulk.hpp
    pika/execution/algorithms/detail/is_negative.hpp
    pika/execution/algorithms/detail/partial_algorithm.hpp
    pika/execution/algorithms/detail/predicates.hpp
    pika/execution/algorithms/detail/single_result.hpp
    pika/execution/algorithms/ensure_started.hpp
    pika/execution/algorithms/execute.hpp
    pika/execution/algorithms/just.hpp
    pika/execution/algorithms/keep_future.hpp
    pika/execution/algorithms/let_error.hpp
    pika/execution/algorithms/let_value.hpp
    pika/execution/algorithms/make_future.hpp
    pika/execution/algorithms/schedule_from.hpp
    pika/execution/algorithms/split.hpp
    pika/execution/algorithms/start_detached.hpp
    pika/execution/algorithms/sync_wait.hpp
    pika/execution/algorithms/then.hpp
    pika/execution/algorithms/transfer.hpp
    pika/execution/algorithms/transfer_just.hpp
    pika/execution/algorithms/when_all.hpp
    pika/execution/detail/async_launch_policy_dispatch.hpp
    pika/execution/detail/execution_parameter_callbacks.hpp
    pika/execution/detail/future_exec.hpp
    pika/execution/detail/post_policy_dispatch.hpp
    pika/execution/detail/sync_launch_policy_dispatch.hpp
    pika/execution/execution.hpp
    pika/execution/executor_parameters.hpp
    pika/execution/executors/auto_chunk_size.hpp
    pika/execution/executors/dynamic_chunk_size.hpp
    pika/execution/executors/execution.hpp
    pika/execution/executors/execution_information.hpp
    pika/execution/executors/execution_parameters.hpp
    pika/execution/executors/execution_parameters_fwd.hpp
    pika/execution/executors/fused_bulk_execute.hpp
    pika/execution/executors/guided_chunk_size.hpp
    pika/execution/executors/persistent_auto_chunk_size.hpp
    pika/execution/executors/polymorphic_executor.hpp
    pika/execution/executors/rebind_executor.hpp
    pika/execution/executors/static_chunk_size.hpp
    pika/execution/traits/detail/simd/vector_pack_alignment_size.hpp
    pika/execution/traits/detail/simd/vector_pack_all_any_none.hpp
    pika/execution/traits/detail/simd/vector_pack_count_bits.hpp
    pika/execution/traits/detail/simd/vector_pack_find.hpp
    pika/execution/traits/detail/simd/vector_pack_load_store.hpp
    pika/execution/traits/detail/simd/vector_pack_type.hpp
    pika/execution/traits/detail/vc/vector_pack_alignment_size.hpp
    pika/execution/traits/detail/vc/vector_pack_all_any_none.hpp
    pika/execution/traits/detail/vc/vector_pack_count_bits.hpp
    pika/execution/traits/detail/vc/vector_pack_find.hpp
    pika/execution/traits/detail/vc/vector_pack_load_store.hpp
    pika/execution/traits/detail/vc/vector_pack_type.hpp
    pika/execution/traits/executor_traits.hpp
    pika/execution/traits/future_then_result_exec.hpp
    pika/execution/traits/is_execution_policy.hpp
    pika/execution/traits/vector_pack_alignment_size.hpp
    pika/execution/traits/vector_pack_all_any_none.hpp
    pika/execution/traits/vector_pack_count_bits.hpp
    pika/execution/traits/vector_pack_find.hpp
    pika/execution/traits/vector_pack_load_store.hpp
    pika/execution/traits/vector_pack_type.hpp
)

set(execution_sources execution_parameter_callbacks.cpp
                      polymorphic_executor.cpp
)

if(TARGET Vc::vc)
  set(execution_optional_dependencies Vc::vc)
endif()

include(pika_add_module)
pika_add_module(
  pika execution
  SOURCES ${execution_sources}
  HEADERS ${execution_headers}
  DEPENDENCIES ${execution_optional_dependencies}
  MODULE_DEPENDENCIES
    pika_async_base
    pika_async_combinators
    pika_config_local
    pika_threading
    pika_pack_traversal
    pika_errors
    pika_memory
    pika_thread_support
    pika_timing
    pika_topology
  CMAKE_SUBDIRS examples tests
)
