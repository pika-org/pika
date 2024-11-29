#!/usr/bin/env bash

# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -euo pipefail

current_dir=$(dirname -- "${BASH_SOURCE[0]}")

source "${current_dir}/utilities.sh"

# Helper function to filter out non-json output, for the type of output that we have below:
# - any number of header lines may be printed that don't consist of '{'
# - the json output starts with a single '{' and continues until the end of the file
# The awk script below prints everything from the first '{' to the end of the file.
function keep_json {
    awk '/^{$/,EOF'
}

metadata_file=$(mktemp --tmpdir metadata.XXXXXXXXXX.json)
create_metadata_file "${metadata_file}"

pika_targets=(
    "task_overhead_report_test"
    "task_size_test"
    "task_size_test"
    "task_size_test"
    "task_size_test"
    "task_size_test"
    "task_latency_test"
    "task_latency_test"
    "task_latency_test"
    "task_latency_test"
    "task_yield_test"
    "task_yield_test"
    "condition_variable_overhead_test"
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

    "--method=task-hierarchical
--tasks-per-thread=1000 \
--task-size-growth-factor=1.05 \
--target-efficiency=0.9 \
--perftest-json"

    "--method=task-yield
--tasks-per-thread=1000 \
--task-size-growth-factor=1.05 \
--target-efficiency=0.9 \
--perftest-json"

    "--method=barrier
--tasks-per-thread=1000 \
--task-size-growth-factor=1.05 \
--target-efficiency=0.9 \
--perftest-json"

    "--method=bulk
--tasks-per-thread=1000 \
--task-size-growth-factor=1.05 \
--target-efficiency=0.5 \
--perftest-json"

    "--repetitions=1000000
--pika:threads=1
--perftest-json"

    "--repetitions=1000000
--nostack
--pika:threads=1
--perftest-json"

    "--repetitions=1000000
--pika:threads=2
--perftest-json"

    "--repetitions=1000000
--nostack
--pika:threads=2
--perftest-json"

    "--repetitions=100
--num-yields=100000
--pika:threads=1
--perftest-json"

    "--repetitions=100
--num-yields=100000
--pika:threads=2
--perftest-json"

    "--loops=1000000
--repetitions=3
--pika:threads=2
--perftest-json"

)

index=0
failures=0
result_file_all=$(mktemp --tmpdir "benchmarks.XXXXXXXXXX.json")
for executable in "${pika_targets[@]}"; do
    test_opts=${pika_test_options[$index]}
    raw_result_file=$(mktemp --tmpdir "${executable}_raw.XXXXXXXXXX.json")
    result_file=$(mktemp --tmpdir "${executable}.XXXXXXXXXX.json")
    echo '{}' >"${result_file}"

    echo
    echo
    echo "Running: ${executable} ${test_opts[@]}"
    set +e
    "${BUILD_DIR}/bin/${executable}" ${test_opts[@]} | tee "${raw_result_file}"
    exit_code=$?
    set -e
    if [[ ${exit_code} -ne 0 ]]; then
        failures=$((failures + 1))

        echo "${executable} failed with exit code ${exit_code}"
    else
        # Append command and command line options
        json_add_value_string "${result_file}" "metric.benchmark.command" "${executable}"
        json_add_value_string "${result_file}" "metric.benchmark.command_options" "${test_opts}"

        # Extract name and timing data from raw result file
        benchmark_name=$(cat "${raw_result_file}" | keep_json | jq '.outputs[0].name')
        benchmark_series=$(cat "${raw_result_file}" | keep_json | jq '.outputs[0].series')
        json_add_value_json "${result_file}" "metric.benchmark.name" "${benchmark_name}"
        json_add_value_json "${result_file}" "metric.benchmark.series" "${benchmark_series}"

        json_merge "${metadata_file}" "${result_file}" "${result_file}"
        cat "${result_file}" >>"${result_file_all}"
    fi

    index=$((index + 1))
done
submit_logstash "${result_file_all}"

if ((failures > 0)); then
    exit 1
fi
