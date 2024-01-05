#!/usr/bin/env bash

# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -euo pipefail

current_dir=$(dirname -- "${BASH_SOURCE[0]}")

source "${current_dir}/json_utilities.sh"

metadata_file=$(mktemp --tmpdir metadata.XXXXXXXXXX.json)
create_metadata_file "${metadata_file}"

pika_targets=(
"task_overhead_report_test"
"task_size_test"
"task_size_test"
"task_size_test"
)
pika_test_options=(
"--pika:ini=pika.thread_queue.init_threads_count=100 \
--pika:queuing=local-priority \
--repetitions=100 \
--tasks=500000"

"--method=task
--tasks-per-thread=1000 \
--task-size-growth-factor=1.05 \
--target-efficiency=0.9 \
--perftest-json"

"--method=barrier
--tasks-per-thread=1000 \
--task-size-growth-factor=1.05 \
--target-efficiency=0.5 \
--perftest-json"

"--method=bulk
--tasks-per-thread=1000 \
--task-size-growth-factor=1.05 \
--target-efficiency=0.5 \
--perftest-json"
)

index=0
for executable in "${pika_targets[@]}"; do
    test_opts=${pika_test_options[$index]}
    raw_result_file=$(mktemp --tmpdir "${executable}_raw.XXXXXXXXXX.json")
    result_file=$(mktemp --tmpdir "${executable}_raw.XXXXXXXXXX.json")
    echo '{}' > "${result_file}"

    "${BUILD_DIR}/bin/${executable}" ${test_opts[@]} > "${raw_result_file}"

    # Append command and command line options
    json_add_value "${result_file}" "metric.benchmark.command" "${executable}"
    json_add_value "${result_file}" "metric.benchmark.command_options" "${test_opts}"

    # Extract name and timing data from raw result file
    benchmark_name=$(jq '.outputs[0].name' "${raw_result_file}")
    benchmark_series=$(jq '.outputs[0].series' "${raw_result_file}")
    json_add_value_json "${result_file}" "metric.benchmark.name" "${benchmark_name}"
    json_add_value_json "${result_file}" "metric.benchmark.series" "${benchmark_series}"

    json_merge "${metadata_file}" "${result_file}" "${result_file}"
    submit_logstash "${result_file}"

    index=$((index + 1))
done
