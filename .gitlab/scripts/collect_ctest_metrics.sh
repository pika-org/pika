#!/usr/bin/env bash

# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -euo pipefail

ctest_xml="${1}"
current_dir=$(dirname -- "${BASH_SOURCE[0]}")

source "${current_dir}/json_utilities.sh"

metadata_file=$(mktemp --tmpdir metadata.XXXXXXXXXX.json)
echo '{}' >"${metadata_file}"

# Logstash data stream metadata section
json_add_value "${metadata_file}" "data_stream.type" "logs"
json_add_value "${metadata_file}" "data_stream.dataset" "service.pika"
json_add_value "${metadata_file}" "data_stream.namespace" "alps"

# CI/git metadata section
json_add_value "${metadata_file}" "ci.organization" "pika-org"
json_add_value "${metadata_file}" "ci.repository" "pika"
json_add_from_env \
    "${metadata_file}" "ci" \
    CI_COMMIT_AUTHOR \
    CI_COMMIT_BRANCH \
    CI_COMMIT_DESCRIPTION \
    CI_COMMIT_MESSAGE \
    CI_COMMIT_SHA \
    CI_COMMIT_SHORT_SHA \
    CI_COMMIT_TIMESTAMP \
    CI_COMMIT_TITLE \
    CI_JOB_IMAGE

# System section
json_add_from_command "${metadata_file}" "system" "hostname"

# Slurm section
json_add_from_env \
    "${metadata_file}" "slurm" \
    SLURM_CLUSTER_NAME \
    SLURM_CPUS_ON_NODE \
    SLURM_CPU_BIND \
    SLURM_JOBID \
    SLURM_JOB_NAME \
    SLURM_JOB_NODELIST \
    SLURM_JOB_NUM_NODES \
    SLURM_JOB_PARTITION \
    SLURM_NODELIST \
    SLURM_NTASKS \
    SLURM_TASKS_PER_NODE

# Build configuration section
json_add_from_env \
    "${metadata_file}" "build_configuration" \
    ARCH \
    BUILD_TYPE \
    CMAKE_COMMON_FLAGS \
    CMAKE_FLAGS \
    COMPILER \
    SPACK_COMMIT \
    SPACK_SPEC

# Submit individual test data
num_tests=$(xq . "${ctest_xml}" | jq '.testsuite.testcase | length')
for i in $(seq 1 ${num_tests}); do
    result_file=$(mktemp --tmpdir "ctest_${i}.XXXXXXXXXX.json")
    echo '{}' >"${result_file}"

    # Extract a single test case from the XML output and convert the time from a string to a number.
    ctest_object=$(xq . "${ctest_xml}" | jq ".testsuite.testcase[$((i - 1))]" | jq '."@time" |= tonumber')
    json_add_value_json "${result_file}" "ctest.testcase" "${ctest_object}"

    json_merge "${metadata_file}" "${result_file}" "${result_file}"
    submit_logstash "${result_file}"
done

result_file=$(mktemp --tmpdir "ctest.XXXXXXXXXX.json")
echo '{}' >"${result_file}"

# Submit overall ctest data. Convert numeric fields from strings to numbers.
ctest_object=$(
    xq . "${ctest_xml}" |
        jq 'del(.testsuite.testcase)' |
        jq '.testsuite."@disabled" |= tonumber' |
        jq '.testsuite."@failures" |= tonumber' |
        jq '.testsuite."@skipped" |= tonumber' |
        jq '.testsuite."@tests" |= tonumber' |
        jq '.testsuite."@time" |= tonumber'
)
json_add_value_json "${result_file}" "ctest" "${ctest_object}"

json_merge "${metadata_file}" "${result_file}" "${result_file}"
submit_logstash "${result_file}"
